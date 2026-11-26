"""
Domain model for the knowledge base.
"""

from __future__ import annotations

from pydantic import model_validator
from .base import Frozen, NonEmptyStr
from .ids import MentionID, RootUID


# If a claim is asserted by the unnamed document author, or if no attribution is specified or inferrable,
# the claimer should be marked as "author" (in example dev KBs) or "default: document author" (in eval KBs).
CLAIMER_AUTHOR_MENTIONS: frozenset[str] = frozenset({"author", "default: document author"})


class KBEntry(Frozen):
    """
    An entry in a document's knowledge base.
    """
    root_uid: RootUID
    ent_id: NonEmptyStr
    entity_mention: NonEmptyStr

    @property
    def is_author(self) -> bool:
        return self.entity_mention in CLAIMER_AUTHOR_MENTIONS


class KBLinking(Frozen):
    qnode_kb_identity_id: NonEmptyStr
    mention_id: MentionID
    root_uid: RootUID
    qnode_kb_type_id: NonEmptyStr | None
    qnode_base_type_id: NonEmptyStr | None
    generic: bool


class DocumentKB(Frozen):
    """
    A knowledge base for a single document.
    """
    root_uid: RootUID
    entries: tuple[KBEntry, ...]

    @model_validator(mode="after")
    def _check(self) -> DocumentKB:
        ids = [e.ent_id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.root_uid}: duplicate ent_id")
        if any(e.root_uid != self.root_uid for e in self.entries):
            raise ValueError(f"{self.root_uid}: entry from another document")
        return self