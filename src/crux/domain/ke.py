"""
Domain model for Knowledge Elements (KEs).
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import Field, model_validator
from typing import Annotated, Literal
from .ids import MentionID, RootUID
from ._base import _Frozen, NonEmptyStr
from .ldc_time import KEDate, KEDateBound


class KEAttribute(StrEnum):
    NOT = "not"
    HEDGED = "hedged"
    IRREALIS = "irrealis"
    GENERIC = "generic"


class RelationKE(_Frozen):
    ke_type: Literal["relation"] = "relation"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]
    start_datetime: KEDate | None = None
    end_datetime: KEDate | None = None

    @model_validator(mode="after")
    def _check(self) -> RelationKE:
        if self.start_datetime and self.start_datetime.bound != KEDateBound.START:
            raise ValueError("start_datetime must have bound 'start'")
        if self.end_datetime and self.end_datetime.bound != KEDateBound.END:
            raise ValueError("end_datetime must have bound 'end'")
        return self


class EventKE(_Frozen):
    ke_type: Literal["event"] = "event"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]
    start_datetime: KEDate | None = None
    end_datetime: KEDate | None = None

    @model_validator(mode="after")
    def _check(self) -> EventKE:
        if self.start_datetime and self.start_datetime.bound != KEDateBound.START:
            raise ValueError("start_datetime must have bound 'start'")
        if self.end_datetime and self.end_datetime.bound != KEDateBound.END:
            raise ValueError("end_datetime must have bound 'end'")
        return self


class ArgumentKE(_Frozen):
    """
    Argument KEs do not have start/end datetime.
    """
    ke_type: Literal["entity"] = "entity"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]


KE = Annotated[
    RelationKE | EventKE | ArgumentKE,
    Field(discriminator="ke_type")
]