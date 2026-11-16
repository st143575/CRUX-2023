from enum import StrEnum
from pydantic import Field
from typing import Annotated, Literal
from .ids import MentionID, RootUID
from ._base import _Frozen


class KEAttribute(StrEnum):
    NOT = "not"
    HEDGED = "hedged"
    IRREALIS = "irrealis"
    GENERIC = "generic"


class RelationKE(_Frozen):
    root_uid: RootUID
    mention_id: MentionID
    ke_type: Literal["relation"]
    attributes: tuple[KEAttribute, ...]


class EventKE(_Frozen):
    root_uid: RootUID
    mention_id: MentionID
    ke_type: Literal["event"]
    attributes: tuple[KEAttribute, ...]


class ArgumentKE(_Frozen):
    root_uid: RootUID
    mention_id: MentionID
    ke_type: Literal["entity"]
    attributes: tuple[KEAttribute, ...]


KE = Annotated[
    RelationKE | EventKE | ArgumentKE, 
    Field(discriminator="ke_type")
]