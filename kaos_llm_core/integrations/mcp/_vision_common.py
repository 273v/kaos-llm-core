"""Shared helpers for the vision MCP tool wrappers.

The three vision tools (``vision_ocr`` / ``vision_describe`` /
``vision_classify``) all take the same page-image input — a filesystem
``path`` or a base64-encoded ``image_base64`` — plus an optional ``model``
override, and all run an async :mod:`kaos_llm_core.vision` program. This module
centralizes the shared parameter schema and the image-loading/validation that
turns the tool inputs into a :class:`~kaos_content.images.model.KaosImage`.

The vision programs and ``KaosImage`` live behind the ``[vision]`` extra
(Pillow + numpy via ``kaos-content[images]``). Imports are lazy so importing
:mod:`kaos_llm_core` never requires Pillow; a missing extra surfaces as an
actionable tool error rather than an import failure.
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING, Any

from kaos_core import ToolResult
from kaos_core.types.parameters import ParameterSchema

if TYPE_CHECKING:
    from kaos_content.images.model import KaosImage

#: Shared input schema for every vision tool: one image source + optional model.
VISION_INPUT_PARAMETERS: list[ParameterSchema] = [
    ParameterSchema(
        name="path",
        type="string",
        description=(
            "Filesystem path to the page image (PNG / JPEG / etc.). Provide "
            "exactly one of 'path' or 'image_base64'."
        ),
        required=False,
    ),
    ParameterSchema(
        name="image_base64",
        type="string",
        description=(
            "Base64-encoded image bytes (no data: URI prefix). Provide exactly "
            "one of 'path' or 'image_base64'. Use this when the image is not on "
            "the server's filesystem."
        ),
        required=False,
    ),
    ParameterSchema(
        name="model",
        type="string",
        description=(
            "Provider-prefixed vision model id (e.g. 'anthropic:claude-haiku-4-5'). "
            "Defaults to kaos-llm-core's DEFAULT_VISION_MODEL."
        ),
        required=False,
    ),
]

VISION_ERROR_HINT = (
    "Provide exactly one of 'path' (a readable image file) or 'image_base64' "
    "(base64-encoded image bytes), and install the vision extra with "
    "'pip install kaos-llm-core[vision]'."
)


def load_page_image(inputs: dict[str, Any]) -> KaosImage | ToolResult:
    """Build a :class:`KaosImage` from tool inputs, or return a ToolResult error.

    Requires exactly one of ``path`` / ``image_base64``. Returns a
    :class:`~kaos_core.ToolResult` error (never raises) for missing/ambiguous
    inputs, a missing ``[vision]`` extra, malformed base64, or an unreadable /
    invalid image. ``KaosImage`` enforces its own decompression-bomb and
    pixel-budget limits on the decoded image.
    """
    path = inputs.get("path")
    image_base64 = inputs.get("image_base64")
    if (path is None) == (image_base64 is None):
        return ToolResult.create_error(
            "vision tools require exactly one of 'path' or 'image_base64'."
        )

    try:
        from kaos_content.images.model import KaosImage
    except ImportError as exc:
        return ToolResult.create_error(
            "vision tools require the [vision] extra (Pillow). Install with "
            f"'pip install kaos-llm-core[vision]'. ({exc})"
        )

    try:
        if path is not None:
            if not isinstance(path, str) or not path:
                return ToolResult.create_error("'path' must be a non-empty string.")
            return KaosImage.from_path(path)
        if not isinstance(image_base64, str) or not image_base64:
            return ToolResult.create_error("'image_base64' must be a non-empty base64 string.")
        try:
            data = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            return ToolResult.create_error(f"'image_base64' is not valid base64: {exc}.")
        if not data:
            return ToolResult.create_error("'image_base64' decoded to empty bytes.")
        return KaosImage.from_bytes(data)
    except (OSError, ValueError) as exc:
        return ToolResult.create_error(f"could not load page image: {exc}.")
