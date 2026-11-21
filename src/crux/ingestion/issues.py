"""
Issues found while ingesting external data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from pydantic import ValidationError


class IssueType(StrEnum):
    ERROR = "error"
    WARNING = "warning"

@dataclass(frozen=True)
class IngestionIssue:
    issue_type: IssueType
    source: str
    location: str
    message: str

    @classmethod
    def error(cls, source: Path | str, location: str, message: str) -> IngestionIssue:
        return cls(IssueType.ERROR, str(source), location, message)

    @classmethod
    def warning(cls, source: Path | str, location: str, message: str) -> IngestionIssue:
        return cls(IssueType.WARNING, str(source), location, message)


def summarize_validation_error(e: ValidationError) -> str:
    """
    Summarize a Pydantic validation error into a concise string.
    """
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or 'model'}: {err['msg']}" for err in e.errors())