"""
Claim date/time: a date/time type plus a partially known timestamp.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from pydantic import Field, model_validator
from ._base import _Frozen


class DatetimeType(StrEnum):
    BEFORE = "before"
    ON = "on"
    AFTER = "after"


_UNKNOWN_TOKENS = frozenset({"unknown", "EMPTY_NA", ""})
_PARTS = ("year", "month", "day", "hour", "minute", "second")
_PATTERN = re.compile(
    r"(before|on|after)\s+"
    r"(\d{4}|[Xx]{4})-(\d{2}|[Xx]{2})-(\d{2}|[Xx]{2})"
    r"T(\d{2}|[Xx]{2}):(\d{2}|[Xx]{2}):(\d{2}|[Xx]{2})"
)


class ClaimDatetime(_Frozen):
    """
    When a claim was made. Components that the document does not give are None.
    A fully unknown claim date/time is represented by None, not by this class.
    """
    type: DatetimeType
    year: int | None = Field(default=None, ge=1, le=9999)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    second: int | None = Field(default=None, ge=0, le=59)

    @model_validator(mode="after")
    def _check(self) -> ClaimDatetime:
        if all(getattr(self, p) is None for p in _PARTS):
            raise ValueError("Claim date/time has no known component; use None for an unknown claim date/time")
        if self.year and self.month and self.day:
            date(self.year, self.month, self.day)
        return self

    @classmethod
    def parse(cls, raw: str) -> ClaimDatetime | None:
        raw = raw.strip()
        if raw in _UNKNOWN_TOKENS:
            return None
        m = _PATTERN.fullmatch(raw)
        if m is None:
            raise ValueError(f"malformed claim date/time: {raw!r}")
        type_, *parts = m.groups()
        values = [None if p[0] in "Xx" else int(p) for p in parts]
        if all(v is None for v in values):
            return None
        return cls(type=DatetimeType(type_), **dict(zip(_PARTS, values)))

    def __str__(self) -> str:
        def part(value: int | None, width: int, x: str) -> str:
            return x * width if value is None else f"{value:0{width}d}"
        ymd = "-".join((part(self.year, 4, "X"), part(self.month, 2, "X"), part(self.day, 2, "X")))
        hms = ":".join(part(v, 2, "x") for v in (self.hour, self.minute, self.second))
        return f"{self.type.value} {ymd}T{hms}"
