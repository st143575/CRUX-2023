from enum import StrEnum
from pydantic import Field
from typing import Annotated, Literal
from .ids import MentionID, RootUID
from ._base import _Frozen, NonEmptyStr


class KEAttribute(StrEnum):
    NOT = "not"
    HEDGED = "hedged"
    IRREALIS = "irrealis"
    GENERIC = "generic"


class RelationKE(_Frozen):
    ke_type: Literal["relation"]
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]


class EventKE(_Frozen):
    ke_type: Literal["event"]
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]


class ArgumentKE(_Frozen):
    ke_type: Literal["entity"]
    root_uid: RootUID
    mention_id: MentionID
    description: str
    qnode_type_ids: tuple[NonEmptyStr, ...]
    attributes: tuple[KEAttribute, ...]


KE = Annotated[
    RelationKE | EventKE | ArgumentKE, 
    Field(discriminator="ke_type")
]