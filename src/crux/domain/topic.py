"""
Domain models for topics, subtopics, and claim templates.
A subtopic (and its corresponding claim template) is assigned a unique ID (e.g., "C303").
Treat topic and subtopic as attributes of a ClaimTemplate object.
"""

from __future__ import annotations

import re
from pydantic import model_validator
from typing import NewType
from ._base import _Frozen, NonEmptyStr


TemplateID = NewType("TemplateID", NonEmptyStr)   # the "ID" field in the LDC2021E11 dataset corresponding to a subtopic and its template, e.g., "C303".
TopicName = NewType("TopicName", NonEmptyStr)     # the "Topic" field in the LDC2021E11 dataset

# "X", "Animal-X", "location/GPE/facility-x", "Government-X" etc.
_X_PLACEHOLDER = re.compile(
    r"(?:(?P<hint>[A-Za-z/]+)-)?(?<![A-Za-z0-9])(?P<x>[Xx])(?![A-Za-z0-9])"
)

class ClaimTemplate(_Frozen):
    template_id: TemplateID
    topic: TopicName
    subtopic: NonEmptyStr
    template_text: NonEmptyStr

    @model_validator(mode="after")
    def _check_placeholder(self) -> ClaimTemplate:
        if len(_X_PLACEHOLDER.findall(self.template_text)) != 1:
            raise ValueError(f"{self.template_id}: template must contain exactly one X: {self.template_text!r}")
        return self

    @property
    def x_hint(self) -> str | None:
        """
        Prefix indicating the type of X, e.g., 'Animal', 'Government', 'date' etc. None if not present.
        """
        m = _X_PLACEHOLDER.search(self.template_text)
        return m["hint"] if m else None

    def populate(self, x: str) -> str:
        """
        Populate the template with the value of X (including the type prefix).
        Used for prompts and natural language descriptions.
        """
        return _X_PLACEHOLDER.sub(lambda _m: x, self.template_text, count=1)