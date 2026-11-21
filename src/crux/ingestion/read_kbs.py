"""
Read per-document KB files into DocumentKB objects.

- One CSV file per document, named <root_uid>.csv, with the columns entID and entity_mention.
- The files are comma-separated, so standard CSV quoting is used: a quoted mention may contain commas.
- ent_ids are local to one file: the same ent_id can mean different entities in different files,
  and one entity can appear under several ent_ids within a file. Duplicate mentions are therefore allowed.
- The document author is "default: document author" (EntID_200) in eval KBs and "author" in the dev
  sample KBs; some dev sample KBs have no author entry at all.
"""

from __future__ import annotations

import csv
from crux.domain.ids import RootUID
from crux.domain.kb import DocumentKB, KBEntry
from crux.ingestion.issues import IngestionIssue, summarize_validation_error
from crux.ingestion.read_documents import index_files
from collections.abc import Iterable
from pathlib import Path
from pydantic import ValidationError


KB_SUFFIX = ".csv"

REQUIRED_COLUMNS: frozenset[str] = frozenset({"entID", "entity_mention"})


def read_kb_file(path: Path, root_uid: RootUID | None = None, ensure_author: bool = False) -> tuple[DocumentKB | None, list[IngestionIssue]]:
    """
    Read a single KB file into a DocumentKB object.
    
    Args:
        path: Path to the KB file
        root_uid: Root UID for the document (optional, can be derived from the filename)
        ensure_author: Whether to add an author entry if none exists
        
    Returns:
        A tuple of (DocumentKB or None, list[IngestionIssue])
    """
    root_uid = root_uid or RootUID(path.name.removesuffix(KB_SUFFIX))
    entries: list[KBEntry] = []
    issues: list[IngestionIssue] = []
    first_line: dict[str, int] = {}
    
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            issues.append(IngestionIssue.error(path, "header", f"missing columns: {sorted(missing)}"))
            return None, issues
        if columns - REQUIRED_COLUMNS:
            issues.append(IngestionIssue.warning(path, "header", f"unexpected columns ignored: {sorted(columns - REQUIRED_COLUMNS)}"))

        for row in reader:
            where = f"line {reader.line_num}"
            if None in row:
                issues.append(IngestionIssue.error(path, where, f"extra fields: {row[None]}"))
                continue

            ent_id = (row["entID"] or "").strip()
            mention = row["entity_mention"] or ""
            if not ent_id or not mention.strip():
                issues.append(IngestionIssue.error(path, where, "empty entID or entity_mention"))
                continue
            if mention != mention.strip():
                # Output must copy KB mentions verbatim; stripping would change them.
                issues.append(IngestionIssue.warning(path, where, f"{ent_id}: surrounding whitespace in mention will be stripped"))
            if ent_id in first_line:
                issues.append(IngestionIssue.error(path, where, f"duplicate entID {ent_id} (first at line {first_line[ent_id]}), skipped"))
                continue
            first_line[ent_id] = reader.line_num

            try:
                entries.append(KBEntry(root_uid=root_uid, ent_id=ent_id, entity_mention=mention))
            except ValidationError as e:
                issues.append(IngestionIssue.error(path, where, summarize_validation_error(e)))

    if not entries:
        issues.append(IngestionIssue.warning(path, "file", "KB has no entries"))

    authors = [e for e in entries if e.is_author]
    if len(authors) > 1:
        issues.append(IngestionIssue.warning(path, "file", f"{len(authors)} author entries: {[e.ent_id for e in authors]}"))
    if not authors and ensure_author:
        entries.append(KBEntry(root_uid=root_uid, ent_id="EntID_200", entity_mention="default: document author"))
        issues.append(IngestionIssue.warning(path, "file", "no author entry; added 'default: document author'"))

    try:
        kb = DocumentKB(root_uid=root_uid, entries=tuple(entries))
    except ValidationError as e:
        issues.append(IngestionIssue.error(path, "file", summarize_validation_error(e)))
        return None, issues

    return kb, issues


def read_kbs(kb_dir: Path, root_uids: Iterable[str] | None = None, ensure_author: bool = False) -> tuple[dict[RootUID, DocumentKB], list[IngestionIssue]]:
    """
    Read KB files from a directory.
    
    Args:
        kb_dir: Directory containing KB files
        root_uids: Optional list of root UIDs to filter by
        ensure_author: Whether to add an author entry if none exists
        
    Returns:
        A tuple of (dict[root_uid -> DocumentKB], list[IngestionIssue])
    """
    index, issues = index_files(kb_dir, KB_SUFFIX)
    
    if root_uids is None:
        wanted = [RootUID(r) for r in sorted(index)]
    else:
        wanted = [RootUID(r) for r in root_uids]
        unrequested = sorted(set(index) - set(wanted))
        if unrequested:
            issues.append(IngestionIssue.warning(kb_dir, "directory", f"{len(unrequested)} KB file(s) not requested: {unrequested}"))

    kbs: dict[RootUID, DocumentKB] = {}
    for root_uid in wanted:
        path = index.get(root_uid)
        if path is None:
            issues.append(IngestionIssue.error(kb_dir, root_uid, f"KB file {root_uid}{KB_SUFFIX} not found"))
            continue
        kb, kb_issues = read_kb_file(path, root_uid, ensure_author)
        issues.extend(kb_issues)
        if kb is not None:
            kbs[root_uid] = kb

    return kbs, issues