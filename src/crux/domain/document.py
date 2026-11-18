"""
Domain model for documents.
"""

from __future__ import annotations

from ._base import _Frozen, NonEmptyStr
from datetime import date
from enum import StrEnum
from pydantic import Field, model_validator
from typing import Annotated, Literal
from .ids import ChildUID, RootUID, SegmentID


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# Videos and audio are out of scope for AIDA Phase 3.
IN_SCOPE_MODALITIES: frozenset[Modality] = frozenset({Modality.TEXT, Modality.IMAGE})

# Documents are harvested from English, Spanish, and Russian media sources.
EVAL_LANGUAGES: frozenset[str] = frozenset({"eng", "spa", "rus"})


class ChildAssetRef(_Frozen):
    """
    Opaque pointer to binary content. 
    Resolved by an AssetLoader port, never by the domain.
    
    Attributes:
        locator: The locator of the child asset, e.g., ".ldcc" path.
        media_type: The media type of the child asset, e.g., "image/jpeg".
        md5: The MD5 hash of the child asset.
    """
    locator: str
    media_type: str | None = None
    md5: str | None = None


class _ChildAssetBase(_Frozen):
    """
    A child asset object represents a child asset of a document, including texts, images, videos, and audio.
    
    Attributes:
        child_uid: The ID of the child asset.
        url: The URL of the child asset.
        asset_type: The type of the child asset file, e.g., ".ltf.xml", ".jpg", ".mp3" etc.
    """
    child_uid: ChildUID
    url: NonEmptyStr
    asset_type: str


class Segment(_Frozen):
    """
    A span of TextChildAsset.text, located by character offsets (end inclusive).
    
    Attributes:
        seg_id: The ID of the segment.
        start: The start index of the segment.
        end: The end index of the segment.

    Note: 
        The LDC annotation uses 0-based indexing with inclusive end. 
        The rsd text may start with a newline ('\n'), so the first segment often has start=1.
        This domain model uses the same indexing for consistency.
    """
    seg_id: SegmentID
    start: int
    end: int


class TextChildAsset(_ChildAssetBase):
    modality: Literal[Modality.TEXT] = Modality.TEXT
    language: NonEmptyStr
    text: str
    segments: tuple[Segment, ...]

    @model_validator(mode="after")
    def _check_segments(self) -> TextChildAsset:
        seen: set[SegmentID] = set()
        for seg in self.segments:
            if not 0 <= seg.start <= seg.end < len(self.text):
                raise ValueError(f"{self.child_uid}: segment {seg.seg_id} out of range")
            if seg.seg_id in seen:
                raise ValueError(f"{self.child_uid}: duplicate segment {seg.seg_id}")
            seen.add(seg.seg_id)
        return self

    def segment_text(self, seg: Segment) -> str:
        return self.text[seg.start : seg.end + 1]


class ImageChildAsset(_ChildAssetBase):
    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    content: ChildAssetRef


class VideoChildAsset(_ChildAssetBase):
    """
    Out of scope for AIDA Phase 3, kept for future development.
    """
    modality: Literal[Modality.VIDEO] = Modality.VIDEO
    content: ChildAssetRef


class AudioChildAsset(_ChildAssetBase):
    """
    Out of scope for AIDA Phase 3, kept for future development.
    """
    modality: Literal[Modality.AUDIO] = Modality.AUDIO
    content: ChildAssetRef

ChildAsset = Annotated[
    TextChildAsset | ImageChildAsset | VideoChildAsset | AudioChildAsset,
    Field(discriminator="modality"),
]

class Document(_Frozen):
    """
    A document object represents the content of a 'root' HTML web page and its associated child assets, including
    texts, images, videos, and audios.

    Attributes:
        root_uid: The ID of the "root" HTML web page, a.k.a. parent_uid.
        content_date: The date the document was created.
        source_url: The URL of the document.
        child_assets: A list of child assets of the document.
    """
    root_uid: RootUID
    content_date: date | None
    source_url: NonEmptyStr | None
    child_assets: tuple[ChildAsset, ...] = ()

    @property
    def text_assets(self) -> tuple[TextChildAsset, ...]:
        return tuple(a for a in self.child_assets if isinstance(a, TextChildAsset))

    @property
    def image_assets(self) -> tuple[ImageChildAsset, ...]:
        return tuple(a for a in self.child_assets if isinstance(a, ImageChildAsset))

    @property
    def video_assets(self) -> tuple[VideoChildAsset, ...]:
        return tuple(a for a in self.child_assets if isinstance(a, VideoChildAsset))

    @property
    def audio_assets(self) -> tuple[AudioChildAsset, ...]:
        return tuple(a for a in self.child_assets if isinstance(a, AudioChildAsset))

    @property
    def language(self) -> NonEmptyStr | None:
        """
        Language of the document, determined by the language of the text asset.
        """
        langs = {a.language for a in self.text_assets if a.language}
        return langs.pop() if len(langs) == 1 else None