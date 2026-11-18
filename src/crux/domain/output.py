"""
Domain model for the system output.
"""

from __future__ import annotations

from pydantic import model_validator
from ._base import _Frozen, NonEmptyStr
from .claim import (
    EpistemicStatus,
    SentimentStatus,
)
from .ids import RootUID, ClaimID
from .kb import KBEntry
from .ldc_time import ClaimDatetime
from .topic import ClaimTemplate


OUTPUT_COLUMNS: tuple[str, ...] = (
    "root_uid", "claim_id", "topic", "claim_template", "x_variable", "claimer",
    "epistemic_status", "claimer_affiliation", "sentiment_status",
    "claim_datetime", "claim_location", "claim_medium",
)

# Annotation guidelines and gold use "author"; KBs may use "default: document author".
AUTHOR_OUTPUT_MENTION = "author"


class OutputClaimFrame(_Frozen):
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

    def to_row(self) -> tuple[str, ...]:
        def _none2str(v: str | None) -> str:
            """Convert None to "EMPTY_NA"."""
            return v if v is not None else "EMPTY_NA"
        
        aff = self.claimer_affiliation.entity_mention if self.claimer_affiliation else None
        claimer = AUTHOR_OUTPUT_MENTION if self.claimer.is_author else self.claimer.entity_mention
        
        return (
            self.root_uid, 
            self.claim_id,
            self.claim_template.topic, 
            self.claim_template.template_text,
            self.x_variable.entity_mention, 
            claimer,
            self.epistemic_status.value, 
            _none2str(aff), 
            self.sentiment_status.value,
            _none2str(str(self.claim_datetime) if self.claim_datetime else None),
            _none2str(self.claim_location.entity_mention if self.claim_location else None),
            _none2str(self.claim_medium.entity_mention if self.claim_medium else None),
        )