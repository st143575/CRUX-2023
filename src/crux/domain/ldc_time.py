"""
Claim datetime: a datetime type plus a partially known timestamp.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from pydantic import Field, model_validator
from ._base import _Frozen


_UNKNOWN_TOKENS = frozenset({"unknown", "EMPTY_NA", ""})
_PARTS = ("year", "month", "day", "hour", "minute", "second")
_PATTERN_CLAIMDATETIME = re.compile(
    r"(before|on|after)\s+"
    r"(\d{4}|[Xx]{4})-(\d{2}|[Xx]{2})-(\d{2}|[Xx]{2})"
    r"T(\d{2}|[Xx]{2}):(\d{2}|[Xx]{2}):(\d{2}|[Xx]{2})"
)
_PATTERN_KEDATE = re.compile(
    r"(start|end)(before|on|after)\s+"
    r"(\d{4}|[Xx]{4})-(\d{2}|[Xx]{2})-(\d{2}|[Xx]{2})"
)

###############################################################################
#                               ClaimDatetime                                 #
###############################################################################

class ClaimDatetimeType(StrEnum):
    BEFORE = "before"
    ON = "on"
    AFTER = "after"

class ClaimDatetime(_Frozen):
    """
    When a claim was made. Components that the document does not give are None.
    A fully unknown claim datetime is represented by None, not by this class.
    """
    type: ClaimDatetimeType
    year: int | None = Field(default=None, ge=1, le=9999)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    second: int | None = Field(default=None, ge=0, le=59)

    @model_validator(mode="after")
    def _check(self) -> ClaimDatetime:
        if all(getattr(self, p) is None for p in _PARTS):
            raise ValueError("Claim datetime has unknown component; use None for an unknown claim datetime")
        if self.year and self.month and self.day:
            date(self.year, self.month, self.day)
        return self

    @classmethod
    def parse(cls, raw: str) -> ClaimDatetime | None:
        raw = raw.strip()
        if raw in _UNKNOWN_TOKENS:
            return None
        m = _PATTERN_CLAIMDATETIME.fullmatch(raw)
        if m is None:
            raise ValueError(f"malformed claim datetime: {raw!r}")
        type_, *parts = m.groups()
        values = [None if p[0] in "Xx" else int(p) for p in parts]
        if all(v is None for v in values):
            return None
        return cls(type=ClaimDatetimeType(type_), **dict(zip(_PARTS, values)))

    def __str__(self) -> str:
        def part(value: int | None, width: int, x: str) -> str:
            return x * width if value is None else f"{value:0{width}d}"
        ymd = "-".join((part(self.year, 4, "x"), part(self.month, 2, "x"), part(self.day, 2, "x")))
        hms = ":".join(part(v, 2, "x") for v in (self.hour, self.minute, self.second))
        return f"{self.type.value} {ymd}T{hms}"


###############################################################################
#                         KEStartDate & KEEndDate                             #
###############################################################################

class KEDateBound(StrEnum):
    START = "start"
    END = "end"

class KEDate(_Frozen):
    bound: KEDateBound
    type: ClaimDatetimeType   # Reuse before / on / after
    year: int | None = Field(default=None, ge=1, le=9999)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _check(self) -> KEDate:
        if all(getattr(self, p) is None for p in ("year", "month", "day")):
            raise ValueError("KEDate has unknown component; use None for an unknown KEDate")
        if self.year and self.month and self.day:
            date(self.year, self.month, self.day)
        return self

    @classmethod
    def parse(cls, raw: str) -> KEDate | None:
        raw = raw.strip()
        if raw in _UNKNOWN_TOKENS or raw.startswith(("startunk", "endunk")):
            return None
        m = _PATTERN_KEDATE.fullmatch(raw)
        if m is None:
            raise ValueError(f"malformed KE date: {raw!r}")
        bound, type_, *parts = m.groups()
        values = [None if p[0] in "Xx" else int(p) for p in parts]
        if all(v is None for v in values):
            return None
        return cls(bound=KEDateBound(bound), type=ClaimDatetimeType(type_), **dict(zip(_PARTS, values)))
