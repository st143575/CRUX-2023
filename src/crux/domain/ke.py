"""
Domain model for Knowledge Elements (KEs).
"""

from enum import StrEnum
from pydantic import Field
from typing import Annotated
from .ids import MentionID, RootUID
from ._base import _Frozen, NonEmptyStr
from .ldc_time import KEStartDate, KEEndDate


class KEAttribute(StrEnum):
    NOT = "not"
    HEDGED = "hedged"
    IRREALIS = "irrealis"
    GENERIC = "generic"


class RelationKE(_Frozen):
    ke_type: "relation"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]
    start_datetime: KEStartDate | None = None
    end_datetime: KEEndDate | None = None


class EventKE(_Frozen):
    ke_type: "event"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]
    start_datetime: KEStartDate | None = None
    end_datetime: KEEndDate | None = None


class ArgumentKE(_Frozen):
    """
    Argument KEs do not have start/end datetime.
    """
    ke_type: "entity"
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]


KE = Annotated[
    RelationKE | EventKE | ArgumentKE,
    Field(discriminator="ke_type")
]