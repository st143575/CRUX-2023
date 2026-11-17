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
    r"(startbefore|starton|startafter|endbefore|endon|endafter)\s+"
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
        ymd = "-".join((part(self.year, 4, "X"), part(self.month, 2, "X"), part(self.day, 2, "X")))
        hms = ":".join(part(v, 2, "x") for v in (self.hour, self.minute, self.second))
        return f"{self.type.value} {ymd}T{hms}"


###############################################################################
#                         KEStartDate & KEEndDate                             #
###############################################################################

class KEStartDateType(StrEnum):
    STARTBEFORE = "startbefore"
    STARTON = "starton"
    STARTAFTER = "startafter"
    STARTUNKNOWN = "startunk"

class KEEndDateType(StrEnum):
    ENDBEFORE = "endbefore"
    ENDON = "endon"
    ENDAFTER = "endafter"
    ENDUNKNOWN = "endunk"

class KEStartDate(_Frozen):
    """
    When an event/relation started. Components that the document does not give are None.
    A fully unknown event/relation date is represented by None, not by this class.
    """
    start_date_type: KEStartDateType
    start_year: int | None = Field(default=None, ge=1, le=9999)
    start_month: int | None = Field(default=None, ge=1, le=12)
    start_day: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _check(self) -> KEStartDate:
        if all(getattr(self, p) is None for p in _PARTS):
            raise ValueError("KE date has unknown component; use None for an unknown KE date")
        if self.start_year and self.start_month and self.start_day:
            date(self.start_year, self.start_month, self.start_day)
        return self

    @classmethod
    def parse(cls, raw: str) -> KEStartDate | None:
        raw = raw.strip()
        if raw in _UNKNOWN_TOKENS:
            return None
        m = _PATTERN_KEDATE.fullmatch(raw)
        if m is None:
            raise ValueError(f"malformed KE start date: {raw!r}")
        type_, *parts = m.groups()
        values = [None if p[0] in "Xx" else int(p) for p in parts]
        if all(v is None for v in values):
            return None
        return cls(start_date_type=KEStartDateType(type_), **dict(zip(_PARTS, values)))

    def __str__(self) -> str:
        def part(value: int | None, width: int, x: str) -> str:
            return x * width if value is None else f"{value:0{width}d}"
        start_ymd = "-".join((part(self.start_year, 4, "X"), part(self.start_month, 2, "X"), part(self.start_day, 2, "X")))
        return f"{self.start_date_type.value} {start_ymd}"

class KEEndDate(_Frozen):
    """
    When an event/relation ended. Components that the document does not give are None.
    A fully unknown event/relation date is represented by None, not by this class.
    """
    end_date_type: KEEndDateType
    end_year: int | None = Field(default=None, ge=1, le=9999)
    end_month: int | None = Field(default=None, ge=1, le=12)
    end_day: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _check(self) -> KEEndDate:
        if all(getattr(self, p) is None for p in _PARTS):
            raise ValueError("KE date has unknown component; use None for an unknown KE date")
        if self.end_year and self.end_month and self.end_day:
            date(self.end_year, self.end_month, self.end_day)
        return self

    @classmethod
    def parse(cls, raw: str) -> KEEndDate | None:
        raw = raw.strip()
        if raw in _UNKNOWN_TOKENS:
            return None
        m = _PATTERN_KEDATE.fullmatch(raw)
        if m is None:
            raise ValueError(f"malformed KE end date: {raw!r}")
        type_, *parts = m.groups()
        values = [None if p[0] in "Xx" else int(p) for p in parts]
        if all(v is None for v in values):
            return None
        return cls(end_date_type=KEEndDateType(type_), **dict(zip(_PARTS, values)))

    def __str__(self) -> str:
        def part(value: int | None, width: int, x: str) -> str:
            return x * width if value is None else f"{value:0{width}d}"
        end_ymd = "-".join((part(self.end_year, 4, "X"), part(self.end_month, 2, "X"), part(self.end_day, 2, "X")))
        return f"{self.end_date_type.value} {end_ymd}"
