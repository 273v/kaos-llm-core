"""Multimodal signature field types.

These are thin subclasses of :class:`kaos_llm_client.types.BinaryData` with
semantic helper methods. They are used as input field types in Signatures::

    from kaos_llm_core import InputField, OutputField, Signature
    from kaos_llm_core.signatures.multimodal import Image

    class Caption(Signature):
        '''Generate an alt-text caption for an image.'''
        image: Image = InputField(description="The image to caption")
        caption: str = OutputField(description="A short alt-text caption")

The codec layer routes ``BinaryData`` inputs through the provider's native
attachment mechanism (``image_url`` for OpenAI, content blocks for Anthropic,
etc.) — see :mod:`kaos_llm_core.codecs.json_codec` and
:mod:`kaos_llm_core.codecs.xml_codec` for the routing logic (Phase 8.2,
§5.2 of the kaos-llm-core roadmap).
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from kaos_llm_client.types import BinaryData


def _base64_from_path(path: str | Path) -> tuple[str, str]:
    """Read a file and return (base64-data, media_type).

    Media type is inferred from the extension; falls back to
    ``application/octet-stream``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Multimodal field file not found: {p}. "
            f"Check the path exists and is readable. "
            f"Use an absolute path if the relative path is ambiguous."
        )
    media_type, _ = mimetypes.guess_type(str(p))
    if media_type is None:
        media_type = "application/octet-stream"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return data, media_type


class Image(BinaryData):
    """An image input to a Signature.

    Subclass of :class:`kaos_llm_client.types.BinaryData`. Codecs detect
    fields whose type is a ``BinaryData`` subclass and route them through
    the provider's native image attachment path.
    """

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """Read a local image file and return an ``Image`` instance.

        Media type is inferred from the file extension. Raises
        ``FileNotFoundError`` with recovery guidance if the file is missing.
        """
        data, media_type = _base64_from_path(path)
        if not media_type.startswith("image/"):
            # Default to png if the extension didn't map to an image type
            media_type = "image/png"
        return cls(data=data, media_type=media_type)

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str = "image/png") -> Image:
        """Build an ``Image`` from raw bytes with an explicit media type."""
        return cls(data=base64.b64encode(raw).decode("ascii"), media_type=media_type)

    @classmethod
    def from_url(cls, url: str) -> Image:
        """Reference an image by URL.

        URL-only images are not currently supported by the multimodal
        signature field layer because ``BinaryData`` requires inline base64.
        Fetch the bytes with ``httpx`` (or similar) and pass them through
        :meth:`from_bytes`. Alternatively, pass a ``data:`` URI via
        :meth:`BinaryData.from_data_uri`.
        """
        raise NotImplementedError(
            "Image.from_url() is not supported because BinaryData requires "
            "inline base64 data. To use a remote image, fetch the bytes "
            "(e.g. with httpx) and call Image.from_bytes(data, media_type). "
            "For data: URIs use Image.from_data_uri()."
        )

    @classmethod
    def from_data_uri(cls, uri: str) -> Image:
        """Parse a ``data:image/...;base64,...`` URI into an ``Image``."""
        bd = BinaryData.from_data_uri(uri)
        return cls(data=bd.data, media_type=bd.media_type)


class Audio(BinaryData):
    """An audio input to a Signature.

    Subclass of :class:`kaos_llm_client.types.BinaryData`. Codecs detect
    audio fields and route them through provider-native audio attachments.
    """

    @classmethod
    def from_path(cls, path: str | Path) -> Audio:
        """Read a local audio file and return an ``Audio`` instance."""
        data, media_type = _base64_from_path(path)
        if not media_type.startswith("audio/"):
            media_type = "audio/wav"
        return cls(data=data, media_type=media_type)

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str = "audio/wav") -> Audio:
        """Build an ``Audio`` from raw bytes with an explicit media type."""
        return cls(data=base64.b64encode(raw).decode("ascii"), media_type=media_type)


class Document(BinaryData):
    """A document input to a Signature (PDF, text, CSV, HTML, markdown).

    Subclass of :class:`kaos_llm_client.types.BinaryData`. Codecs detect
    document fields and route them through provider-native document
    attachments (Anthropic ``document`` blocks, OpenAI file parts, etc.).
    """

    @classmethod
    def from_path(cls, path: str | Path) -> Document:
        """Read a local document file and return a ``Document`` instance."""
        data, media_type = _base64_from_path(path)
        return cls(data=data, media_type=media_type)

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str = "application/pdf") -> Document:
        """Build a ``Document`` from raw bytes with an explicit media type."""
        return cls(data=base64.b64encode(raw).decode("ascii"), media_type=media_type)


def is_binary_field(annotation: Any) -> bool:
    """Return True if a type annotation is a ``BinaryData`` subclass.

    Used by codecs to detect multimodal fields and route them through
    provider-native attachment mechanisms.
    """
    try:
        return isinstance(annotation, type) and issubclass(annotation, BinaryData)
    except TypeError:
        return False


__all__ = [
    "Audio",
    "Document",
    "Image",
    "is_binary_field",
]
