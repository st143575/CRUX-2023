"""
Build Document objects from parent_children.tab, LTF files and RSD files.

- parent_children.tab: one row per child asset; groups children under their root (parent_uid).
- <child_uid>.ltf.xml: segment offsets (0-based, inclusive end) and the language of a text asset.
- <child_uid>.rsd.txt: the raw text that the LTF offsets point into.

Images, videos and audio are recorded as references only; their content is never read here.
"""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from crux.domain.document import (
    AudioChildAsset,
    ChildAsset,
    ChildAssetRef,
    Document, 
    ImageChildAsset,
    Modality, 
    Segment, 
    TextChildAsset,
    VideoChildAsset,
)
from crux.domain.ids import ChildUID, RootUID, SegmentID
from crux.ingestion.issues import IngestionIssue, _summarize
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from pydantic import ValidationError


# Required columns in parent_children.tab
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "parent_uid", "child_uid", "url", "child_asset_type", "lang_id", "lang_manual",
    "rel_pos", "unwrapped_md5", "content_date", "status_in_corpus",
})

TEXT_ASSET_TYPE = ".ltf.xml"

# parent_children.tab writes missing values as "n/a"; treat them like empty cells.
NULL_TOKENS: frozenset[str] = frozenset({"", "n/a"})

# File extension (after removing ".ldcc") -> (modality, media type)
_MEDIA: dict[str, tuple[Modality, str]] = {
    # image files
    "bmp": (Modality.IMAGE, "image/bmp"),
    "gif": (Modality.IMAGE, "image/gif"),
    "jpg": (Modality.IMAGE, "image/jpeg"),
    "png": (Modality.IMAGE, "image/png"),
    "svg": (Modality.IMAGE, "image/svg+xml"),
    # audio files
    "mp3": (Modality.AUDIO, "audio/mpeg"),
    # video files
    "mp4": (Modality.VIDEO, "video/mp4"),
}

@dataclass(frozen=True)
class ChildRow:
    """
    One row of parent_children.tab, kept as close to the file as possible.
    """
    line: int
    parent_uid: str
    child_uid: str
    url: str | None
    child_asset_type: str
    lang: str | None            # lang_manual if present, else lang_id; None for non-text assets
    rel_pos: float | None
    unwrapped_md5: str | None
    content_date: str | None
    status: str | None


###############################################################################
#                            parent_children.tab                              #
###############################################################################

def read_parent_children(path: Path, delimiter: str = "\t") -> tuple[dict[RootUID, list[ChildRow]], list[IngestionIssue]]:
    """
    Group the rows of parent_children.tab by root UID.
    
    Args:
        path: Path to the parent_children.tab file.
        delimiter: The delimiter used in the file (default is tab).
        
    Returns:
        A tuple of a dictionary mapping root UIDs to lists of ChildRow objects and a list of ingestion issues.
    """
    groups: dict[RootUID, list[ChildRow]] = defaultdict(list)
    issues: list[IngestionIssue] = []
    
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            issues.append(IngestionIssue.error(path, "header", f"missing columns: {sorted(missing)}"))
            return {}, issues

        for row in reader:
            where = f"line {reader.line_num}"
            if None in row:
                issues.append(IngestionIssue.error(path, where, f"extra fields: {row[None]}"))
                continue
            
            values = {k: _opt(v) for k, v in row.items()}

            parent_uid, child_uid, asset_type = values["parent_uid"], values["child_uid"], values["child_asset_type"]
            if parent_uid is None:
                issues.append(IngestionIssue.error(path, where, "empty parent_uid"))
                continue

            if child_uid is None:
                if asset_type == TEXT_ASSET_TYPE:
                    msg = "text asset not available (child_uid is n/a)"
                else:
                    msg = f"{asset_type or 'asset'} without child_uid, skipped"
                issues.append(IngestionIssue.warning(path, where, f"{parent_uid}: {msg}"))
                continue
            
            if asset_type is None:
                issues.append(IngestionIssue.error(path, where, f"{child_uid}: empty child_asset_type"))
                continue

            rel_pos: float | None = None
            if values["rel_pos"]:
                try:
                    rel_pos = float(values["rel_pos"])
                except ValueError:  # keep the row (no continue): rel_pos only affects ordering
                    issues.append(IngestionIssue.warning(path, where, f"invalid rel_pos {values['rel_pos']!r}"))
            
            groups[RootUID(parent_uid)].append(ChildRow(
                line=reader.line_num,
                parent_uid=parent_uid,
                child_uid=child_uid,
                url=values["url"],
                child_asset_type=asset_type,
                lang=values["lang_manual"] or values["lang_id"],
                rel_pos=rel_pos,
                unwrapped_md5=values["unwrapped_md5"],
                content_date=values["content_date"],
                status=values["status_in_corpus"],
            ))

    return dict(groups), issues


###############################################################################
#                                 LTF + RSD                                   #
###############################################################################

def index_files(directory: Path, suffix: str) -> tuple[dict[str, Path], list[IngestionIssue]]:
    """
    Map <uid> to the file path for every <uid><suffix> in the directory.
    This facilitates recursive lookup of LTF and RSD files that may be nested in subdirectories.
    
    Args:
        directory: Directory to search
        suffix: File suffix to look for (e.g., ".ltf.xml")
        
    Returns:
        Tuple of (dict[uid -> Path], list[IngestionIssue])
    """
    index: dict[str, Path] = {}
    issues: list[IngestionIssue] = []
    for path in sorted(directory.rglob(f"*{suffix}")):
        uid = path.name.removesuffix(suffix)
        if uid in index:
            issues.append(IngestionIssue.error(path, "file", f"duplicate file for {uid}; also at {index[uid]}"))
            continue
        index[uid] = path

    return index, issues


def read_text_asset(row: ChildRow, ltf_index: Mapping[str, Path], rsd_index: Mapping[str, Path]) -> tuple[TextChildAsset | None, list[IngestionIssue]]:
    """
    Build a TextChildAsset from LTF and RSD files.

    The asset is dropped (None) if the RSD text does not match the LTF exactly,
    because every segment offset would then point to the wrong character.
    
    Args:
        row: ChildRow with child_uid and other metadata
        ltf_index: Mapping of child_uid to LTF file path
        rsd_index: Mapping of child_uid to RSD file path
        
    Returns:
        Tuple of (TextChildAsset, list[IngestionIssue])
    """
    ltf_path = ltf_index.get(row.child_uid)
    rsd_path = rsd_index.get(row.child_uid)
    
    if ltf_path is None or rsd_path is None:
        missing = [name for name, path in (("ltf", ltf_path), ("rsd", rsd_path)) if path is None]
        return None, [IngestionIssue.error(row.child_uid, "file", f"{' and '.join(missing)} file not found")]
    
    issues: list[IngestionIssue] = []

    # Read LTF file.
    try:
        doc_el = ET.parse(ltf_path).getroot().find("DOC")
    except ET.ParseError as e:
        return None, [IngestionIssue.error(ltf_path, "file", f"XML parse error: {e}")]

    # Check DOC element.
    if doc_el is None:
        return None, [IngestionIssue.error(ltf_path, "file", "DOC element not found")]
    
    # Check DOC id (should match child_uid).
    if doc_el.get("id") != row.child_uid:
        issues.append(IngestionIssue.error(ltf_path, "DOC", f"DOC id mismatch: expected {row.child_uid}, got {doc_el.get('id')}"))

    # Read RSD file.
    try:
        with rsd_path.open("r", encoding="utf-8", newline="") as f:     # newline="": keep "\r\n" as is, otherwise every offset after it would shift
            rsd_text = f.read()
    except UnicodeDecodeError as e:
        return None, issues + [IngestionIssue.error(rsd_path, "file", f"not valid UTF-8: {e}")]
    
    # Check raw_text_char_length.
    expected_len = _int_or_none(doc_el.get("raw_text_char_length"))
    if expected_len is not None and expected_len != len(rsd_text):
        issues.append(IngestionIssue.error(rsd_path, "file", f"raw_text_char_length mismatch: expected {expected_len}, got {len(rsd_text)}"))

    # Check raw_text_md5.
    expected_md5 = doc_el.get("raw_text_md5")
    if expected_md5 is not None and hashlib.md5(rsd_text.encode("utf-8")).hexdigest() != expected_md5:
        issues.append(IngestionIssue.error(rsd_path, "file", "md5 does not match raw_text_md5 in the LTF"))

    if issues:
        return None, issues

    # Read segments from LTF.
    segments: list[Segment] = []
    for seg_el in doc_el.iter("SEG"):
        start, end = _int_or_none(seg_el.get("start_char")), _int_or_none(seg_el.get("end_char"))
        if start is None or end is None:
            issues.append(IngestionIssue.error(ltf_path, f"SEG {seg_el.get('id')}", "missing or non-integer start_char/end_char"))
            continue
        # The offsets must point exactly at ORIGINAL_TEXT; otherwise every evidence citation would be shifted.
        # XML parsers turn "\r\n" and "\r" into "\n" inside ORIGINAL_TEXT while the RSD keeps them,
        # so line endings are normalised for this comparison only.
        original = seg_el.findtext("ORIGINAL_TEXT") or ""
        if _newlines(rsd_text[start:end + 1]) != _newlines(original):
            issues.append(IngestionIssue.error(ltf_path, f"SEG {seg_el.get('id')}", f"offsets ({start}, {end}) do not match ORIGINAL_TEXT"))
            continue
        # LTF segment ids (e.g., "segment-0") are only unique within an LTF file.
        # Make them globally unique by prefixing with the child UID (i.e., <child_uid>:<segment_id>).
        segments.append(Segment(seg_id=SegmentID(f"{row.child_uid}:{seg_el.get('id')}"), start=start, end=end))

    if issues:
        return None, issues

    # Check language.
    language = doc_el.get("lang") or row.lang
    if language is None:
        issues.append(IngestionIssue.error(ltf_path, "DOC", "no language specified in LTF or parent_children"))
        return None, issues
    if row.lang and language != row.lang:
        issues.append(IngestionIssue.warning(ltf_path, "DOC", f"language mismatch: LTF has {language}, parent_children has {row.lang}"))
    
    # Check url.
    if row.url is None:
        issues.append(IngestionIssue.error(ltf_path, "DOC", "no url specified in parent_children"))
        return None, issues

    try:
        asset = TextChildAsset(
            child_uid=ChildUID(row.child_uid),
            url=row.url,
            asset_type=row.child_asset_type,
            language=language,
            text=rsd_text,
            segments=tuple(segments)
        )
    except ValidationError as e:
        issues.append(IngestionIssue.error(ltf_path, "asset", _summarize(e)))
        return None, issues
    
    return asset, issues



###############################################################################
#                                 Documents                                   #
###############################################################################

def build_document(
    root_uid: RootUID,
    rows: list[ChildRow],
    ltf_index: Mapping[str, Path],
    rsd_index: Mapping[str, Path],
    source: str = "parent_children.tab"
) -> tuple[Document | None, list[IngestionIssue]]:
    """
    Assemble a single Document from a list of ChildRow objects.
    
    Args:
        root_uid: The root UID of the document.
        rows: List of ChildRow objects.
        ltf_index: Mapping of child_uid to LTF file path.
        rsd_index: Mapping of child_uid to RSD file path.
        source: Source of the document (default: "parent_children.tab").
    
    Returns:
        A tuple of (Document, list[IngestionIssue]) where Document is None if there are errors.
    """
    assets: list[ChildAsset] = []
    issues: list[IngestionIssue] = []
    seen: set[str] = set()

    for row in sorted(rows, key=lambda r: (r.rel_pos is None, r.rel_pos or 0.0, r.child_uid)):
        where = f"line {row.line}"
        
        if row.child_uid in seen:
            issues.append(IngestionIssue.error(source, where, f"duplicate child_uid {row.child_uid}"))
            continue
        seen.add(row.child_uid)

        if row.url is None:
            issues.append(IngestionIssue.error(source, where, f"{row.child_uid}: empty url"))
            continue

        if row.status != "present":
            issues.append(IngestionIssue.warning(source, where, f"child_uid {row.child_uid}: status_in_corpus={row.status!r}, skipped"))
            continue
        
        if row.child_asset_type == TEXT_ASSET_TYPE:
            asset, asset_issues = read_text_asset(row, ltf_index, rsd_index)
            issues.extend(asset_issues)
            if asset:
                assets.append(asset)
            continue

        media = _MEDIA.get(row.child_asset_type.removesuffix(".ldcc").lstrip("."))
        if media is None:
            issues.append(IngestionIssue.error(source, where, f"unknown child_asset_type {row.child_asset_type!r}, skipped"))
            continue    
        modality, media_type = media

        ref = ChildAssetRef(
            locator=f"{row.child_uid}{row.child_asset_type}",
            media_type=media_type, 
            md5=row.unwrapped_md5 or None
        )

        cls = {
            Modality.IMAGE: ImageChildAsset,
            Modality.AUDIO: AudioChildAsset,
            Modality.VIDEO: VideoChildAsset,
        }[modality]

        try:
            assets.append(cls(child_uid=ChildUID(row.child_uid), url=row.url, asset_type=row.child_asset_type, content=ref))
        except ValidationError as e:
            issues.append(IngestionIssue.error(source, where, _summarize(e)))

    if not any(isinstance(a, TextChildAsset) for a in assets):
        issues.append(IngestionIssue.warning(source, root_uid, "document has no usable text asset"))

    try:
        doc = Document(
            root_uid=root_uid,
            content_date=_content_date(root_uid, rows, source, issues),
            source_url=_source_url(root_uid, rows, source, issues),
            child_assets=tuple(assets),
        )
    except ValidationError as e:
        issues.append(IngestionIssue.error(source, root_uid, _summarize(e)))
        return None, issues
    
    return doc, issues


def read_documents(
    parent_children_path: Path,
    ltf_dir: Path,
    rsd_dir: Path,
    root_uids: Iterable[str] | None = None,
    delimiter: str = "\t",
) -> tuple[tuple[Document, ...], list[IngestionIssue]]:
    """
    Read all documents, or only the specified root UIDs, in the order requested.
    
    Args:
        parent_children_path: Path to parent_children.tab
        ltf_dir: Directory containing LTF files
        rsd_dir: Directory containing RSD files
        root_uids: Optional list of root UIDs to filter by
        delimiter: delimiter used to parse tabular input (default: tab)
    
    Returns:
        Tuple of (documents, issues)
    """
    groups, issues = read_parent_children(parent_children_path, delimiter)
    ltf_index, ltf_issues = index_files(ltf_dir, ".ltf.xml")
    rsd_index, rsd_issues = index_files(rsd_dir, ".rsd.txt")
    issues += ltf_issues + rsd_issues
    if root_uids is None:
        wanted = list(groups)
    else:
        wanted = [RootUID(r) for r in root_uids]
    
    documents: list[Document] = []
    for root_uid in wanted:
        rows = groups.get(root_uid)
        if rows is None:
            issues.append(IngestionIssue.error(parent_children_path, root_uid, f"root_uid {root_uid} not found in parent_children.tab"))
            continue
        doc, doc_issues = build_document(root_uid, rows, ltf_index, rsd_index, source=str(parent_children_path))
        issues.extend(doc_issues)
        if doc is not None:
            documents.append(doc)
    
    return tuple(documents), issues


###############################################################################
#                                  Helpers                                    #
###############################################################################

def _opt(value: str | None) -> str | None:
    """Strip a cell and map the null tokens ("", "n/a") to None."""
    value = (value or "").strip()
    return None if value in NULL_TOKENS else value


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _newlines(s: str) -> str:
    """
    Normalise line endings, for comparing RSD text with ORIGINAL_TEXT.
    """
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _content_date(root_uid: RootUID, rows: list[ChildRow], source: str, issues: list[IngestionIssue]) -> date | None:
    raw = {r.content_date for r in rows if r.content_date}
    if not raw:
        return None
    if len(raw) > 1:
        issues.append(IngestionIssue.warning(source, root_uid, f"conflicting content_date values {sorted(raw)}"))
    preferred = next(
        (r.content_date for r in rows if r.content_date and r.child_asset_type == TEXT_ASSET_TYPE),
        sorted(raw)[0],
    )
    try:
        return date.fromisoformat(preferred)
    except ValueError:
        issues.append(IngestionIssue.warning(source, root_uid, f"unparseable content_date {preferred!r}"))
        return None


def _source_url(root_uid: RootUID, rows: list[ChildRow], source: str, issues: list[IngestionIssue]) -> str | None:
    urls = Counter(r.url for r in rows if r.url)
    if not urls:
        return None
    if len(urls) > 1:
        issues.append(IngestionIssue.warning(source, root_uid, f"child assets of document {root_uid} have {len(urls)} different urls"))
    return urls.most_common(1)[0][0]