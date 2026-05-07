"""Tool primitive — a (ToolDefinition, executor) pair for tool-calling Programs.

Part of ``kaos-llm-core`` 0.1.0. This module is intentionally tiny: it exists
so that tool-calling programs (``ReAct``, ``BestOfN``, future planners) can
accept a uniform input type that carries both the wire schema the model sees
and the Python callable that actually runs when the model selects the tool.

Design rules pinned in the §5.1 roadmap decision:

- A ``Tool`` is a **thin pair**. It holds exactly two things: a
  ``ToolDefinition`` (the wire schema, owned by ``kaos-llm-client``) and an
  ``executor`` callable. It does **not** carry its own schema fields and it
  does **not** reinvent tool calling.
- ``Tool.from_callable()`` is the only place where schema is generated. It
  builds a ``ToolDefinition`` from the callable's type hints and docstring.
- Runtime argument validation is the provider's job (structured output
  already enforces the JSON Schema). The executor may add any extra
  runtime checks it wants.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kaos_core.logging import get_logger
from kaos_llm_client.types import ToolDefinition
from pydantic import BaseModel

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool a Program can pass to an LLM that supports tool calling.

    A ``Tool`` is a thin pair of:

    - ``definition``: the wire schema sent to the model (a ``ToolDefinition``
      from ``kaos-llm-client``). ``kaos-llm-core`` never reinvents tool
      schemas — ``definition`` is the single source of truth.
    - ``executor``: the Python callable invoked when the model selects this
      tool. May be sync or async; ``await`` works in both cases via the
      :meth:`invoke` helper.

    Construct a ``Tool`` either directly (when you already have a
    ``ToolDefinition``) or via :meth:`Tool.from_callable` (which derives a
    ``ToolDefinition`` from the callable's type hints and docstring).
    """

    definition: ToolDefinition
    executor: Callable[..., Any] | Callable[..., Awaitable[Any]]

    @property
    def name(self) -> str:
        """The tool name as declared on the underlying ``ToolDefinition``."""
        return self.definition.name

    @property
    def description(self) -> str | None:
        """The tool description as declared on the underlying ``ToolDefinition``."""
        return self.definition.description

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Call the executor with model-provided arguments.

        Sync executors are called directly; async executors are awaited.
        Argument validation is intentionally NOT performed here — the
        provider's structured output already validates against
        ``definition.parameters``. The executor is responsible for any
        additional runtime checks.
        """
        result = self.executor(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> Tool:
        """Build a Tool by introspecting a Python callable.

        Generates a ``ToolDefinition`` whose:

        - name = ``name`` arg or ``fn.__name__``
        - description = ``description`` arg or ``fn.__doc__`` (first paragraph)
        - parameters = JSON Schema derived from ``fn``'s type hints

        Type-hint-to-JSON-Schema mapping uses :func:`typing.get_type_hints`
        for Python 3.14 compatibility. Supported types: ``str``, ``int``,
        ``float``, ``bool``, ``list[T]``, ``dict``, ``Optional[T]``, plus
        Pydantic ``BaseModel`` subclasses (which use ``model_json_schema()``).

        Parameters without type hints raise ``TypeError``. Parameters named
        ``self`` or ``cls`` are skipped so methods and classmethods work.
        Parameters with defaults are optional; all others are required.

        Raises:
            TypeError: If a parameter has no type hint or an unsupported type.
        """
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            raise TypeError(
                "Tool.from_callable: callable has no __name__ and no explicit name was "
                "provided. Pass `name=` or build a Tool with an explicit ToolDefinition."
            )

        tool_description = description if description is not None else _first_paragraph(fn.__doc__)

        # typing.get_type_hints is the Python 3.14-safe way to resolve
        # deferred annotations. Fallback to the raw __annotations__ map if
        # get_type_hints cannot resolve (e.g., forward refs that depend on
        # a scope the caller did not pass in).
        try:
            hints = get_type_hints(fn)
        except Exception as exc:
            logger.debug(
                "Tool.from_callable: get_type_hints failed for %r, "
                "falling back to __annotations__: %s",
                tool_name,
                exc,
            )
            hints = dict(getattr(fn, "__annotations__", {}))
        hints.pop("return", None)

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Tool.from_callable: could not inspect signature of '{tool_name}': {exc}. "
                "Wrap the callable in a plain function, or build a Tool with an "
                "explicit ToolDefinition."
            ) from exc

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            # Skip bound-method receivers so ``Tool.from_callable(obj.method)``
            # and classmethod descriptors both work.
            if param_name in ("self", "cls"):
                continue
            # Skip variadic *args / **kwargs — they have no meaningful
            # JSON-Schema projection.
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            if param_name not in hints:
                raise TypeError(
                    f"Tool.from_callable: parameter '{param_name}' on '{tool_name}' has "
                    "no type hint. Add a type hint or build a Tool with an explicit "
                    "ToolDefinition."
                )

            annotation = hints[param_name]
            schema, is_optional = _annotation_to_schema(annotation, tool_name, param_name)
            properties[param_name] = schema

            has_default = param.default is not inspect.Parameter.empty
            if not has_default and not is_optional:
                required.append(param_name)

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters_schema,
            strict=strict,
        )
        return cls(definition=definition, executor=fn)


# ---------------------------------------------------------------------------
# Helpers — deliberately private
# ---------------------------------------------------------------------------


def _first_paragraph(docstring: str | None) -> str | None:
    """Return the first paragraph of a docstring (text before the first blank line)."""
    if not docstring:
        return None
    text = inspect.cleandoc(docstring)
    if not text:
        return None
    paragraph, _, _ = text.partition("\n\n")
    stripped = paragraph.strip()
    return stripped or None


_PRIMITIVE_MAP: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    bytes: {"type": "string", "format": "byte"},
}


def _annotation_to_schema(
    annotation: Any, tool_name: str, param_name: str
) -> tuple[dict[str, Any], bool]:
    """Convert a Python type annotation to a (JSON Schema, is_optional) tuple.

    ``is_optional`` is True when the annotation is ``T | None`` / ``Optional[T]``;
    in that case the returned schema describes ``T`` and the caller should
    treat the parameter as not-required.
    """
    origin = get_origin(annotation)

    # Optional[T] / T | None
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        non_none = tuple(a for a in args if a is not type(None))
        if len(non_none) == 1 and len(non_none) != len(args):
            # Exactly one non-None arm — treat as Optional[T]
            inner_schema, _ = _annotation_to_schema(non_none[0], tool_name, param_name)
            return inner_schema, True
        # Multi-arm unions fall through to the unsupported-type error below.
        raise TypeError(
            f"Tool.from_callable: parameter '{param_name}' on '{tool_name}' uses an "
            f"unsupported union type {annotation!r}. Supported: primitives, list[T], "
            "dict, Optional[T], and Pydantic BaseModel subclasses. Collapse the union "
            "to a single type or build a Tool with an explicit ToolDefinition."
        )

    # list[T]
    if origin in (list, tuple):
        args = get_args(annotation)
        if not args:
            return {"type": "array"}, False
        item_schema, _ = _annotation_to_schema(args[0], tool_name, param_name)
        return {"type": "array", "items": item_schema}, False

    # dict / dict[str, Any]
    if origin is dict:
        return {"type": "object"}, False
    if annotation is dict:
        return {"type": "object"}, False

    # Primitive scalars
    if isinstance(annotation, type):
        if annotation in _PRIMITIVE_MAP:
            return dict(_PRIMITIVE_MAP[annotation]), False

        # Pydantic BaseModel subclass — use its own JSON Schema
        if issubclass(annotation, BaseModel):
            return annotation.model_json_schema(), False

    raise TypeError(
        f"Tool.from_callable: parameter '{param_name}' on '{tool_name}' has "
        f"unsupported type {annotation!r}. Supported: str, int, float, bool, "
        "list[T], dict, Optional[T], and Pydantic BaseModel subclasses. "
        "Add a supported type hint or build a Tool with an explicit ToolDefinition."
    )
