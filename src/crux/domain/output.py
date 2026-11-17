"""
Domain model for the system output.
"""

from ._base import _Frozen, NonEmptyStr
from .ids import RootUID, ClaimID
from .claim import EpistemicStatus, SentimentStatus


class OutputClaimFrame(_Frozen):
    """
    A claim frame provides information about the core claim, the claimer and their affiliations,
    the epistemic and sentiment status of the claim, and the datetime, location and medium of the claim.
    """
    root_uid: RootUID
    claim_id: ClaimID
    topic: NonEmptyStr
    subtopic: NonEmptyStr
    claim_template: NonEmptyStr
    x_variable: NonEmptyStr
    claimer: NonEmptyStr
    claimer_affiliation: NonEmptyStr | "EMPTY_NA"
    epistemic_status: EpistemicStatus
    sentiment_status: SentimentStatus
    claim_datetime: NonEmptyStr | "EMPTY_NA"
    claim_location: NonEmptyStr | "EMPTY_NA"
    claim_medium: NonEmptyStr | "EMPTY_NA"