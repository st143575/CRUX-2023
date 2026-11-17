"""
Domain model for the knowledge base.
"""

from ._base import _Frozen, NonEmptyStr
from .ids import MentionID, RootUID


class KB(_Frozen):
    entID: NonEmptyStr
    entity_mention: NonEmptyStr


class KBLinking(_Frozen):
    qnode_kb_identity_id: NonEmptyStr
    mention_id: MentionID
    root_uid: RootUID
    qnode_kb_type_ids: NonEmptyStr | None
    qnode_base_type_id: NonEmptyStr | None
    generic: bool