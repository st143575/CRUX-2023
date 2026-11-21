"""
Read the topic_list.txt file provided by LDC2021E11 into ClaimTemplate objects.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from crux.domain.topic import ClaimTemplate
from crux.ingestion.issues import IngestionIssue, _summarize
from pathlib import Path
from pydantic import ValidationError


# A mapping from topic_list.txt column names to ClaimTemplate field names.
DEFAULT_COLUMNS: dict[str, str] = {
    "ID": "template_id",
    "subtopic": "subtopic",
    "topic": "topic",
    "Template": "template_text",
}


def read_topic_list(path: Path, columns: dict[str, str] = DEFAULT_COLUMNS) -> tuple[tuple[ClaimTemplate, ...], list[IngestionIssue]]:
    """
    Read the topic_list.txt file provided by LDC2021E11 into ClaimTemplate objects.
    Parse each row, collect all issues, and let the caller decide how to handle them.
    
    Args:
        path: Path to the topic_list.txt file.
        columns: A mapping from topic_list.txt column names to ClaimTemplate field names.
        
    Returns:
        A tuple of ClaimTemplate objects and a list of ingestion issues.
    """
    templates: list[ClaimTemplate] = []
    issues: list[IngestionIssue] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            issues.append(IngestionIssue.error(path, "header", f"missing columns: {sorted(missing)}"))
            return (), issues

        for row in reader:
            where = f"line {reader.line_num}"
            if None in row:
                issues.append(IngestionIssue.error(path, where, f"extra fields: {row[None]}"))
                continue
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                templates.append(ClaimTemplate.model_validate({field: row[col] for col, field in columns.items()}))
            except ValidationError as e:
                issues.append(IngestionIssue.error(path, where, _summarize(e)))
    
    issues.extend(_check_unique(path, templates, "template_id", lambda t: t.template_id))
    issues.extend(_check_unique(path, templates, "(topic, subtopic)", lambda t: (t.topic, t.subtopic)))
    issues.extend(_check_unique(path, templates, "(topic, template_text)", lambda t: (t.topic, t.template_text)))

    if not templates and not issues:
        issues.append(IngestionIssue.error(path, "file", "no templates found"))
    
    return tuple(templates), issues


def _check_unique(
    path: Path,
    templates: list[ClaimTemplate],
    what: str,
    key: Callable[[ClaimTemplate], Hashable]
) -> list[IngestionIssue]:
    counts = Counter(key(t) for t in templates)
    return [
        IngestionIssue.error(path, "file", f"duplicate {what}: {k!r} ({n} times)")
        for k, n in counts.items() if n > 1
    ]


def index_by_text(templates: Iterable[ClaimTemplate]) -> dict[tuple[str, str], ClaimTemplate]:
    """
    (topic, template_text) -> ClaimTemplate; used when reading gold annotations.
    Uniqueness is guaranteed by read_topic_list.
    """
    return {(t.topic, t.template_text): t for t in templates}