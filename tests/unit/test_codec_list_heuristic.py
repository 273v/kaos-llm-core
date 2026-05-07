"""Tier 2.7: list-typed-output coercion in ChatCodec / XMLCodec.

The Phase 16.3 codec regression matrix surfaced that ChatCodec and
XMLCodec failed on list-typed output fields because models return
``"Acme, Beta, Gamma"`` instead of ``["Acme", "Beta", "Gamma"]``.
The Phase 16.5 fix adds a ``_coerce_to_list`` post-decode pass that
splits on common delimiters (newlines, semicolons, commas) and
coerces element types (str / int / float) when the field annotation
is ``list[X]``.
"""

from __future__ import annotations

from kaos_llm_core.codecs.chat_codec import _coerce_to_list


class TestCoerceToListString:
    def test_comma_separated(self) -> None:
        assert _coerce_to_list("Acme, Beta, Gamma", list[str]) == [
            "Acme",
            "Beta",
            "Gamma",
        ]

    def test_semicolon_separated(self) -> None:
        assert _coerce_to_list("Acme; Beta; Gamma", list[str]) == [
            "Acme",
            "Beta",
            "Gamma",
        ]

    def test_newline_separated(self) -> None:
        assert _coerce_to_list("Acme\nBeta\nGamma", list[str]) == [
            "Acme",
            "Beta",
            "Gamma",
        ]

    def test_strips_brackets(self) -> None:
        assert _coerce_to_list("[Acme, Beta]", list[str]) == ["Acme", "Beta"]

    def test_strips_quotes(self) -> None:
        assert _coerce_to_list('"Acme", "Beta"', list[str]) == ["Acme", "Beta"]

    def test_single_element(self) -> None:
        assert _coerce_to_list("Acme", list[str]) == ["Acme"]

    def test_empty_string(self) -> None:
        assert _coerce_to_list("", list[str]) == []

    def test_only_whitespace(self) -> None:
        assert _coerce_to_list("   \n  ", list[str]) == []

    def test_drops_empty_pieces(self) -> None:
        assert _coerce_to_list("Acme,, Beta, , Gamma,", list[str]) == [
            "Acme",
            "Beta",
            "Gamma",
        ]


class TestCoerceToListInt:
    def test_comma_separated_ints(self) -> None:
        assert _coerce_to_list("1, 2, 3", list[int]) == [1, 2, 3]

    def test_newline_separated_ints(self) -> None:
        assert _coerce_to_list("1\n2\n3", list[int]) == [1, 2, 3]

    def test_invalid_int_returns_none(self) -> None:
        # Mixed list with a non-numeric piece should fall through.
        assert _coerce_to_list("1, foo, 3", list[int]) is None


class TestCoerceToListFloat:
    def test_comma_separated_floats(self) -> None:
        result = _coerce_to_list("1.5, 2.5, 3.5", list[float])
        assert result == [1.5, 2.5, 3.5]

    def test_invalid_float_returns_none(self) -> None:
        assert _coerce_to_list("1.0, two, 3.0", list[float]) is None


class TestCoerceToListNonListAnnotation:
    def test_returns_none_for_str_annotation(self) -> None:
        assert _coerce_to_list("Acme, Beta", str) is None

    def test_returns_none_for_int_annotation(self) -> None:
        assert _coerce_to_list("42", int) is None

    def test_returns_none_for_dict_annotation(self) -> None:
        assert _coerce_to_list("a: 1", dict[str, int]) is None


# ---------------------------------------------------------------------------
# End-to-end: ChatCodec.decode handles list-typed fields with the heuristic
# ---------------------------------------------------------------------------


class TestChatCodecListDecode:
    def test_chat_codec_decodes_csv_list(self) -> None:
        from kaos_llm_client import ProviderResponse, UsageInfo
        from kaos_llm_client.types import ContentPart

        from kaos_llm_core.codecs.chat_codec import ChatCodec
        from kaos_llm_core.signatures import InputField, OutputField, Signature

        class _Sig(Signature):
            """Extract entities."""

            text: str = InputField(description="Input")
            entities: list[str] = OutputField(description="List of entity names")

        # Model returned the list as a CSV string under the marker.
        response = ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text="[entities]\nAcme Corp, Beta LLC, Gamma Inc.",
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=0, output_tokens=0, total_tokens=0),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )
        decoded = ChatCodec().decode(_Sig, response)
        assert decoded["entities"] == ["Acme Corp", "Beta LLC", "Gamma Inc."]


class TestXMLCodecListDecode:
    def test_xml_codec_decodes_csv_list(self) -> None:
        from kaos_llm_client import ProviderResponse, UsageInfo
        from kaos_llm_client.types import ContentPart

        from kaos_llm_core.codecs.xml_codec import XMLCodec
        from kaos_llm_core.signatures import InputField, OutputField, Signature

        class _Sig(Signature):
            """Extract entities."""

            text: str = InputField(description="Input")
            entities: list[str] = OutputField(description="List of entity names")

        response = ProviderResponse.model_construct(
            provider="function",
            model="function-test",
            raw={},
            parts=[
                ContentPart.model_construct(
                    type="text",
                    text="<entities>Acme Corp, Beta LLC, Gamma Inc.</entities>",
                )
            ],
            usage=UsageInfo.model_construct(input_tokens=0, output_tokens=0, total_tokens=0),
            stop_reason="end_turn",
            status_code=200,
            response_headers={},
        )
        decoded = XMLCodec().decode(_Sig, response)
        assert decoded["entities"] == ["Acme Corp", "Beta LLC", "Gamma Inc."]
