"""
Domain model for the knowledge base.
"""

from ._base import _Frozen, NonEmptyStr
from .ids import MentionID, RootUID


class KBEntry(_Frozen):
    doc_id: NonEmptyStr
    ent_id: NonEmptyStr
    entity_mention: NonEmptyStr


class KBLinking(_Frozen):
    qnode_kb_identity_id: NonEmptyStr
    mention_id: MentionID
    root_uid: RootUID
    qnode_kb_type_id: NonEmptyStr | None
    qnode_base_type_id: NonEmptyStr | None
    generic: bool