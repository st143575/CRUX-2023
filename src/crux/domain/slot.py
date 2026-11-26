"""
Domain model for KE slots.
"""

from .base import Frozen, NonEmptyStr
from .ids import MentionID, RootUID
from typing import Annotated, Literal
from pydantic import Field


class RelationSlot(Frozen):
    slot_type: Literal["relation"] = "relation"
    root_uid: RootUID
    relationmention_id: MentionID
    general_slot_type: NonEmptyStr      # Value can be 'A0', 'A1', or 'EMPTY_TBD'.
    argmention_id: MentionID


class EventSlot(Frozen):
    slot_type: Literal["event"] = "event"
    root_uid: RootUID
    eventmention_id: MentionID
    general_slot_type: NonEmptyStr      # One of 101 candidate values (see tmp/check_data_sample.ipynb).
    argmention_id: MentionID


Slot = Annotated[
    RelationSlot | EventSlot,
    Field(discriminator="slot_type")
]
