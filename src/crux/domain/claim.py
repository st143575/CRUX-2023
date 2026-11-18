"""
Domain model for claim frames.
"""

from __future__ import annotations

from enum import StrEnum
from pydantic import Field, model_validator
from .topic import ClaimTemplate
from ._base import _Frozen, NonEmptyStr
from .ids import MentionID, RootUID, ClaimID
from .ldc_time import ClaimDatetime


class XVariable(_Frozen):
    """
    The item that fills the X in the claim template, as well as its identity Qnode and type Qnodes.
    """
    name: NonEmptyStr                                                 # Value is a string and cannot be empty.
    qnode_identity_id: NonEmptyStr                                    # Value can be a single DWD Qnode ID or NIL Qnode ID and cannot be empty.
    qnode_type_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)     # Value can be one or more DWD Qnode IDs and cannot be empty.


class ClaimerAffiliation(_Frozen):
    """
    A claimer's affiliation, consisting of a name, an identity Qnode, and type Qnodes.
    """
    name: NonEmptyStr                          # Value can be a string or "EMPTY_NA" (the ClaimerAffiliation object is not instantiated at all).
    qnode_identity_id: NonEmptyStr | None      # Value can be a single DWD Qnode ID or NIL Qnode ID, or "EMPTY_NA" (set to None).
    qnode_type_ids: tuple[NonEmptyStr, ...]    # Value can be one or more DWD Qnode IDs, or "EMPTY_NA" (set to empty tuple).


class Claimer(_Frozen):
    """
    A claimer, consisting of a name, an identity Qnode, type Qnodes, a provenance, and up to 3 affiliations.
    """
    name: NonEmptyStr                                                                 # Value is a string and cannot be empty.
    qnode_identity_id: NonEmptyStr                                                    # Value can be a single DWD Qnode ID or NIL Qnode ID and cannot be empty.
    qnode_type_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)                     # Value can be one or more DWD Qnode IDs and cannot be empty.
    provenance: NonEmptyStr | None                                                    # Value can be a string or "EMPTY_NA" (set to None).
    affiliations: tuple[ClaimerAffiliation, ...] = Field(default=(), max_length=3)    # Value can be up to 3 claimer affiliations.


class EpistemicStatus(StrEnum):
    """
    Claimer's stance on the truth and certainty of the core claim.
    """
    TRUE_CERTAIN = "true-certain"
    TRUE_UNCERTAIN = "true-uncertain"
    FALSE_CERTAIN = "false-certain"
    FALSE_UNCERTAIN = "false-uncertain"
    UNKNOWN = "unknown"


class SentimentStatus(StrEnum):
    """
    Claimer's sentiment towards the core claim.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral-unknown"


class ClaimLocation(_Frozen):
    """
    The place where the claim was made, consisting of a name, an identity Qnode, type Qnodes, and a provenance.
    """
    name: NonEmptyStr                         # Value can be a string or "EMPTY_NA" (the ClaimLocation object is not instantiated at all).
    qnode_identity_id: NonEmptyStr | None     # Value can be a single DWD Qnode ID or NIL Qnode ID, or "EMPTY_NA" (set to None).
    qnode_type_ids: tuple[NonEmptyStr, ...]   # Value can be one or more DWD Qnode IDs, or "EMPTY_NA" (set to empty tuple).
    provenance: NonEmptyStr | None            # Value can be a string or "EMPTY_NA" (set to None).


class ClaimMedium(_Frozen):
    """
    The broadcast source on where the claim was asserted, consisting of a name, an identity Qnode, type Qnodes, and a provenance.
    """
    name: NonEmptyStr                         # Value can be a string or "EMPTY_NA" (the ClaimMedium object is not instantiated at all).
    qnode_identity_id: NonEmptyStr | None     # Value can be a single DWD Qnode ID or NIL Qnode ID, or "EMPTY_NA" (set to None).
    qnode_type_ids: tuple[NonEmptyStr, ...]   # Value can be one or more DWD Qnode IDs, or "EMPTY_NA" (set to empty tuple).
    provenance: NonEmptyStr | None            # Value can be a string or "EMPTY_NA" (set to None).


class AnnotatedClaimFrame(_Frozen):
    """
    A claim frame provides information about the core claim, the claimer and their affiliations,
    the epistemic and sentiment status of the claim, and the datetime, location and medium of the claim.

    A claim frame consists of 29 fields. However, this class only includes 15 fields as some of the fields
    (the qnode_identity, qnodes_type, provenance and affiliation fields) are encapsulated in other classes.
    """
    root_uid: RootUID
    claim_id: ClaimID
    # topic: NonEmptyStr        # topic and subtopic are encapsulated in ClaimTemplate
    # subtopic: NonEmptyStr
    claim_template: ClaimTemplate
    x_variable: XVariable
    description: NonEmptyStr
    claim_semantics: tuple[MentionID, ...] = Field(min_length=1, max_length=3)
    claimer: Claimer
    epistemic_status: EpistemicStatus
    sentiment_status: SentimentStatus
    claim_datetime: ClaimDatetime | None
    claim_location: ClaimLocation | None
    claim_medium: ClaimMedium | None
    associated_kes: tuple[MentionID, ...]

    @model_validator(mode="after")
    def _check_semantics_in_associated_kes(self) -> AnnotatedClaimFrame:
        missing = set(self.claim_semantics) - set(self.associated_kes)
        if missing:
            raise ValueError(f"{self.claim_id}: claim semantics not in associated KEs: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _check_semantics_types(self) -> AnnotatedClaimFrame:
        for ke in self.claim_semantics:
            if not ke.startswith("VM") and not ke.startswith("RM"):
                raise ValueError(f"{self.claim_id}: claim semantics must be event KEs (VM) or relation KEs (RM): {ke}")
        return self

    @model_validator(mode="after")
    def _check_associated_kes_types(self) -> AnnotatedClaimFrame:
        for ke in self.associated_kes:
            if not ke.startswith("VM") and not ke.startswith("RM"):
                raise ValueError(f"{self.claim_id}: associated KEs must be event KEs (VM) or relation KEs (RM): {ke}")
        return self

    @property
    def topic(self) -> str:
        return self.claim_template.topic
    
    @property
    def subtopic(self) -> str:
        return self.claim_template.subtopic
