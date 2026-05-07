"""Unit tests for the Tool primitive (kaos-llm-core 0.1.0, Phase 5.1).

Covers direct construction, schema derivation via ``Tool.from_callable()``,
sync/async ``invoke()``, and error cases for unsupported types or missing
annotations. All tests are sync except those that exercise async executors;
``asyncio_mode = "auto"`` in pyproject.toml lets async tests run without
explicit ``@pytest.mark.asyncio`` decorators.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_llm_client.types import ToolDefinition
from pydantic import BaseModel

from kaos_llm_core.programs.tool import Tool


class _QueryModel(BaseModel):
    """Module-level Pydantic model for introspection tests.

    Defined at module scope so ``typing.get_type_hints()`` can resolve the
    annotation when the function that uses it lives inside a test method
    (nested-scope types are not in ``fn.__globals__``).
    """

    q: str
    limit: int = 10


class TestDirectConstruction:
    def test_construct_with_definition_and_callable(self) -> None:
        def echo(msg: str) -> str:
            return msg

        definition = ToolDefinition(
            name="echo-tool",
            description="Echoes a message.",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
                "additionalProperties": False,
            },
        )
        tool = Tool(definition=definition, executor=echo)

        assert tool.definition is definition
        assert tool.executor is echo
        assert tool.name == "echo-tool"
        assert tool.description == "Echoes a message."


class TestFromCallableSimpleTypes:
    def test_simple_primitives(self) -> None:
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        tool = Tool.from_callable(add)
        params = tool.definition.parameters
        assert params["type"] == "object"
        assert params["properties"]["a"] == {"type": "integer"}
        assert params["properties"]["b"] == {"type": "integer"}
        assert sorted(params["required"]) == ["a", "b"]
        assert params["additionalProperties"] is False

    def test_all_primitive_types(self) -> None:
        def f(s: str, i: int, fl: float, b: bool) -> None:
            return None

        tool = Tool.from_callable(f)
        props = tool.definition.parameters["properties"]
        assert props["s"] == {"type": "string"}
        assert props["i"] == {"type": "integer"}
        assert props["fl"] == {"type": "number"}
        assert props["b"] == {"type": "boolean"}


class TestFromCallableOptionalFields:
    def test_parameter_with_default_is_optional(self) -> None:
        def f(a: int, b: str = "x") -> str:
            return f"{a}{b}"

        tool = Tool.from_callable(f)
        params = tool.definition.parameters
        assert params["required"] == ["a"]
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert params["properties"]["b"] == {"type": "string"}

    def test_optional_type_is_not_required(self) -> None:
        def f(a: int | None) -> int:
            return a or 0

        tool = Tool.from_callable(f)
        params = tool.definition.parameters
        # Optional[int] → schema keeps integer type, parameter not required
        assert params["properties"]["a"] == {"type": "integer"}
        assert params["required"] == []

    def test_optional_with_default_none(self) -> None:
        def f(a: str | None = None) -> str:
            return a or ""

        tool = Tool.from_callable(f)
        params = tool.definition.parameters
        assert params["properties"]["a"] == {"type": "string"}
        assert params["required"] == []


class TestFromCallableCollectionTypes:
    def test_list_of_strings(self) -> None:
        def f(items: list[str]) -> int:
            return len(items)

        tool = Tool.from_callable(f)
        params = tool.definition.parameters
        assert params["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert params["required"] == ["items"]

    def test_list_of_ints(self) -> None:
        def f(nums: list[int]) -> int:
            return sum(nums)

        tool = Tool.from_callable(f)
        assert tool.definition.parameters["properties"]["nums"] == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_dict_parameter(self) -> None:
        def f(payload: dict) -> None:
            return None

        tool = Tool.from_callable(f)
        assert tool.definition.parameters["properties"]["payload"] == {"type": "object"}

    def test_dict_str_any_parameter(self) -> None:
        def f(payload: dict[str, Any]) -> None:
            return None

        tool = Tool.from_callable(f)
        assert tool.definition.parameters["properties"]["payload"] == {"type": "object"}


class TestFromCallablePydanticModel:
    def test_pydantic_basemodel_uses_model_json_schema(self) -> None:
        def search(query: _QueryModel) -> list[str]:
            return [query.q]

        tool = Tool.from_callable(search)
        params = tool.definition.parameters
        # The "query" property schema should match _QueryModel.model_json_schema()
        assert params["properties"]["query"] == _QueryModel.model_json_schema()
        assert params["required"] == ["query"]


class TestFromCallableNameAndDescription:
    def test_name_derived_from_fn_name(self) -> None:
        def my_search(q: str) -> str:
            """Search the index."""
            return q

        tool = Tool.from_callable(my_search)
        assert tool.name == "my_search"
        assert tool.description == "Search the index."

    def test_name_override(self) -> None:
        def my_search(q: str) -> str:
            return q

        tool = Tool.from_callable(my_search, name="search")
        assert tool.name == "search"

    def test_description_first_paragraph_only(self) -> None:
        def f(a: int) -> int:
            """First paragraph only.

            Second paragraph should be ignored.
            Third line of second paragraph.
            """
            return a

        tool = Tool.from_callable(f)
        assert tool.description == "First paragraph only."

    def test_description_override(self) -> None:
        def f(a: int) -> int:
            """Original docstring."""
            return a

        tool = Tool.from_callable(f, description="Overridden description.")
        assert tool.description == "Overridden description."

    def test_no_docstring_no_description(self) -> None:
        def f(a: int) -> int:
            return a

        tool = Tool.from_callable(f)
        assert tool.description is None


class TestFromCallableErrors:
    def test_missing_annotation_raises_type_error(self) -> None:
        def broken(a, b: int) -> int:  # type: ignore[no-untyped-def]
            return b

        with pytest.raises(TypeError) as excinfo:
            Tool.from_callable(broken)
        msg = str(excinfo.value)
        assert "'a'" in msg
        assert "broken" in msg
        assert "type hint" in msg

    def test_unsupported_type_raises_type_error(self) -> None:
        class NotAModel:
            pass

        def f(x: NotAModel) -> None:
            return None

        with pytest.raises(TypeError) as excinfo:
            Tool.from_callable(f)
        assert "unsupported type" in str(excinfo.value).lower()

    def test_multi_arm_union_raises(self) -> None:
        def f(x: int | str) -> None:
            return None

        with pytest.raises(TypeError) as excinfo:
            Tool.from_callable(f)
        assert "union" in str(excinfo.value).lower()


class TestFromCallableSkipsSelf:
    def test_method_self_is_skipped(self) -> None:
        class Service:
            def fetch(self, url: str) -> str:
                """Fetch a URL."""
                return url

        svc = Service()
        # Bound method — signature.parameters does not include 'self'
        bound_tool = Tool.from_callable(svc.fetch)
        params = bound_tool.definition.parameters
        assert "self" not in params["properties"]
        assert params["properties"]["url"] == {"type": "string"}
        assert params["required"] == ["url"]

        # Unbound function — signature.parameters includes 'self', we skip it
        unbound_tool = Tool.from_callable(Service.fetch)
        uparams = unbound_tool.definition.parameters
        assert "self" not in uparams["properties"]
        assert uparams["properties"]["url"] == {"type": "string"}
        assert uparams["required"] == ["url"]


class TestInvoke:
    async def test_sync_executor(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool.from_callable(add)
        result = await tool.invoke({"a": 2, "b": 3})
        assert result == 5

    async def test_async_executor(self) -> None:
        async def fetch(url: str) -> str:
            return f"body:{url}"

        tool = Tool.from_callable(fetch)
        result = await tool.invoke({"url": "https://example.com"})
        assert result == "body:https://example.com"

    async def test_mixed_args_expansion(self) -> None:
        def greet(name: str, greeting: str = "hello") -> str:
            return f"{greeting}, {name}"

        tool = Tool.from_callable(greet)
        assert await tool.invoke({"name": "world"}) == "hello, world"
        assert await tool.invoke({"name": "world", "greeting": "hi"}) == "hi, world"


class TestProperties:
    def test_name_and_description_delegate_to_definition(self) -> None:
        def f(a: int) -> int:
            """A docstring."""
            return a

        tool = Tool.from_callable(f)
        # Mutating definition.name should be reflected via the property (same
        # underlying object); verify delegation rather than caching.
        assert tool.name == tool.definition.name
        assert tool.description == tool.definition.description
