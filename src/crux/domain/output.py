"""
Domain model for the system output.
"""

from __future__ import annotations

from pydantic import model_validator
from typing import NamedTuple
from .base import Frozen, NonEmptyStr
from .claim import (
    EpistemicStatus,
    SentimentStatus,
)
from .ids import RootUID, ClaimID
from .kb import KBEntry
from .ldc_time import ClaimDatetime
from .missing_value import none2empty_na
from .topic import ClaimTemplate


class OutputClaimFrameFields(NamedTuple):
    root_uid: str
    claim_id: str
    topic: str
    claim_template: str
    x_variable: str
    claimer: str
    epistemic_status: str
    claimer_affiliation: str
    sentiment_status: str
    claim_datetime: str
    claim_location: str
    claim_medium: str

# The 12 output field names for Task 1.
OUTPUT_COLUMNS: tuple[str, ...] = OutputClaimFrameFields._fields

# Annotation guidelines and gold use "author"; KBs may use "default: document author".
AUTHOR_OUTPUT_MENTION = "author"


class OutputClaimFrame(Frozen):
    """
    A claim frame provides information about the core claim, the claimer and their affiliations,
    the epistemic and sentiment status of the claim, and the datetime, location and medium of the claim.
    """
    root_uid: RootUID
    claim_id: ClaimID
    # topic: NonEmptyStr                            # topic (and subtopic) are encapsulated in ClaimTemplate
    claim_template: ClaimTemplate
    x_variable: KBEntry
    claimer: KBEntry
    epistemic_status: EpistemicStatus
    claimer_affiliation: KBEntry | None             # Value must be an entity mention in a per-document KB; None must be mapped to "EMPTY_NA"
    sentiment_status: SentimentStatus
    claim_datetime: ClaimDatetime | None
    claim_location: KBEntry | None                  # Value must be an entity mention in a per-document KB; None must be mapped to "EMPTY_NA"
    claim_medium: KBEntry | None                    # Value must be an entity mention in a per-document KB; None must be mapped to "EMPTY_NA"

    @model_validator(mode="after")
    def _check_kb_entries(self) -> OutputClaimFrame:
        for e in (self.x_variable, self.claimer, self.claimer_affiliation,
                self.claim_location, self.claim_medium):
            if e is not None and e.root_uid != self.root_uid:
                raise ValueError(f"{self.claim_id}: KB entry {e.ent_id} belongs to {e.root_uid}")
        return self

    @property
    def subtopic(self) -> NonEmptyStr:
        return self.claim_template.subtopic

    def to_row(self) -> OutputClaimFrameFields:
        affiliation = self.claimer_affiliation.entity_mention if self.claimer_affiliation else None
        claimer = AUTHOR_OUTPUT_MENTION if self.claimer.is_author else self.claimer.entity_mention
        
        return OutputClaimFrameFields(
            self.root_uid,
            self.claim_id,
            self.claim_template.topic,
            self.claim_template.template_text,
            self.x_variable.entity_mention,
            claimer,
            self.epistemic_status.value,
            none2empty_na(affiliation),
            self.sentiment_status.value,
            none2empty_na(str(self.claim_datetime) if self.claim_datetime else None),
            none2empty_na(self.claim_location.entity_mention if self.claim_location else None),
            none2empty_na(self.claim_medium.entity_mention if self.claim_medium else None),
        )