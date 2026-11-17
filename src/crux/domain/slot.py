"""
Domain model for KE slots.
"""

from ._base import _Frozen, NonEmptyStr
from .ids import MentionID, RootUID
from typing import Annotated
from pydantic import Field


class RelationSlot(_Frozen):
    slot_type: "relation"
    root_uid: RootUID
    relationmention_id: MentionID
    general_slot_type: NonEmptyStr      # Value can be 'A0', 'A1', or 'EMPTY_TBD'.
    argmention_id: MentionID


class EventSlot(_Frozen):
    slot_type: "event"
    root_uid: RootUID
    eventmention_id: MentionID
    general_slot_type: NonEmptyStr      # One of 101 candidate values (see tmp/check_data_sample.ipynb).
    argmention_id: MentionID


Slot = Annotated[
    RelationSlot | EventSlot,
    Field(discriminator="slot_type")
]
