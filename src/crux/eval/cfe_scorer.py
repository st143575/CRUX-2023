from __future__ import annotations

import numpy as np
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from crux.domain.kb import CLAIMER_AUTHOR_MENTIONS
from crux.domain.ldc_time import ClaimDatetime
from crux.domain.missing_value import EMPTY_NA, EMPTY_TOKENS
from crux.domain.output import AUTHOR_OUTPUT_MENTION, OUTPUT_COLUMNS, OutputClaimFrameFields
from crux.eval.base import PRF
from crux.ingestion.issues import IngestionIssue, IssueType
from crux.ingestion.read_output import read_output_claim_frames
from dataclasses import dataclass, replace
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from types import MappingProxyType


Row = OutputClaimFrameFields

# Field weights for evaluating Task 1: Claim Frame Extraction
CLAIM_FIELDS_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "topic": 0.19,
    "claim_template": 0.19,
    "x_variable": 0.19,
    "claimer": 0.19,
    "epistemic_status": 0.16,
    "claimer_affiliation": 0.02,
    "sentiment_status": 0.02,
    "claim_datetime": 0.02,
    "claim_location": 0.01,
    "claim_medium": 0.01,
})

# A candidate pair scores 0 unless all of these fields match
GATE_FIELDS: tuple[str, ...] = ("topic", "claim_template", "x_variable")


###############################################################################
#                               Normalization                                 #
###############################################################################

def normalize_claim_datetime(value: str) -> str:
    """
    Canonical form of a claim datetime: lower-case "x" placeholders and 
    EMPTY_NA for a fully unknown datetime ("unknown", "EMPTY_NA", "", or all components unknown).
    A malformed value is kept as it is, so that it can still be compared rather than crash the scorer.
    """
    try:
        dt = ClaimDatetime.parse(value)
    except ValueError:
        return value.strip()
    return str(dt) if dt is not None else EMPTY_NA


def normalize_row(row: Row) -> Row:
    """
    Map different spellings of the same value to one form:
    - surrounding whitespace is stripped from every field;
    - "" and EMPTY_NA become EMPTY_NA;
    - claim_datetime is canonicalized by normalize_claim_datetime;
    - the document author ("author" / "default: document author") becomes "author" in claimer.
    """
    values: dict[str, str] = {}
    for name, raw in zip(Row._fields, row):
        value = raw.strip()
        if name == "claim_datetime":
            value = normalize_claim_datetime(value)
        elif value in EMPTY_TOKENS:
            value = EMPTY_NA
        elif name == "claimer" and value in CLAIMER_AUTHOR_MENTIONS:
            value = AUTHOR_OUTPUT_MENTION   # "author"
        values[name] = value
    return Row(**values)


###############################################################################
#                                   Report                                    #
###############################################################################

@dataclass(frozen=True)
class AlignedPair:
    """
    A system claim frame aligned with a gold claim frame with non-zero weight (i.e. a TP).

    Attributes:
        root_uid: The root UID of the document.
        pred_claim_id: The claim ID of the system claim frame.
        gold_claim_id: The claim ID of the gold claim frame.
        score: The scaled weight of the alignment. Value in (0, 1].
    """
    root_uid: str
    pred_claim_id: str
    gold_claim_id: str
    score: float


@dataclass(frozen=True)
class CFEReport:
    """
    Scores for Task 1.

    Attributes:
        claims: Claim-level counts (primary metric).
        weighted_accuracy: Mean scaled weight of the TP pairs.
        fields: Per-field set-based P/R/F1 (secondary metric).
        alignments: The TP pairs, for error analysis.
        n_pred_unparsed: System rows that could not be read and were counted as FP (lenient mode only).
    """
    claims: PRF
    weighted_accuracy: float
    fields: Mapping[str, PRF]
    alignments: tuple[AlignedPair, ...]
    n_pred_unparsed: int = 0

    @property
    def precision(self) -> float:
        return self.claims.precision

    @property
    def recall(self) -> float:
        return self.claims.recall

    @property
    def f1(self) -> float:
        return self.claims.f1

    def to_dict(self) -> dict[str, object]:
        return {
            **self.claims.to_dict(),
            "weighted_accuracy": self.weighted_accuracy,
            "n_pred_unparsed": self.n_pred_unparsed,
            "fields": {name: prf.to_dict() for name, prf in self.fields.items()},
        }


###############################################################################
#                                   Scorer                                    #
###############################################################################

@dataclass(frozen=True)
class CFEScorer:
    """
    Scorer for Task 1: Claim Frame Extraction.

    Attributes:
        weights: Weight of each scored field.
        gate_fields: Fields that must all match for a pair to score above 0.
        normalize: Applied to every system and gold row before comparison; None compares raw strings.
        ignore_empty_na: Leave EMPTY_NA values out of the per-field sets of the secondary metric.
    """
    weights: Mapping[str, float] = CLAIM_FIELDS_WEIGHTS
    gate_fields: tuple[str, ...] = GATE_FIELDS
    normalize: Callable[[Row], Row] | None = normalize_row
    ignore_empty_na: bool = True

    def __post_init__(self) -> None:
        scorable = set(OUTPUT_COLUMNS) - {"root_uid", "claim_id"}
        unknown = (set(self.weights) | set(self.gate_fields)) - scorable
        if unknown:
            raise ValueError(f"not scorable claim frame fields: {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()) or self.total_weight <= 0:
            raise ValueError(f"weights must be non-negative with a positive total: {dict(self.weights)}")

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    def pair_score(self, pred: Row, gold: Row) -> float:
        """
        Scaled weight of a candidate pair.
        Both rows are expected to be already normalized.
        """
        if any(getattr(pred, f) != getattr(gold, f) for f in self.gate_fields):
            return 0.0
        matched = sum(w for f, w in self.weights.items() if getattr(pred, f) == getattr(gold, f))
        return matched / self.total_weight

    def score(self, pred: Sequence[Row], gold: Sequence[Row], /) -> CFEReport:
        if self.normalize is not None:
            pred = [self.normalize(r) for r in pred]
            gold = [self.normalize(r) for r in gold]

        pred_by_doc, gold_by_doc = _group_by_doc(pred), _group_by_doc(gold)
        alignments: list[AlignedPair] = []
        for root_uid in sorted(pred_by_doc.keys() & gold_by_doc.keys()):
            alignments.extend(self._align(root_uid, pred_by_doc[root_uid], gold_by_doc[root_uid]))

        accuracy = sum(a.score for a in alignments) / len(alignments) if alignments else 0.0
        return CFEReport(
            claims=PRF(tp=len(alignments), n_pred=len(pred), n_gold=len(gold)),
            weighted_accuracy=accuracy,
            fields={f: self._field_prf(pred, gold, f) for f in self.weights},
            alignments=tuple(alignments),
        )

    def _align(self, root_uid: str, preds: Sequence[Row], golds: Sequence[Row]) -> list[AlignedPair]:
        """
        Maximum-weight bipartite matching within one document; keep only pairs with non-zero weight.
        """
        weights = np.array([[self.pair_score(p, g) for g in golds] for p in preds])
        rows, cols = linear_sum_assignment(weights, maximize=True)
        return [
            AlignedPair(root_uid, preds[i].claim_id, golds[j].claim_id, float(weights[i, j]))
            for i, j in zip(rows, cols)
            if weights[i, j] > 0
        ]

    def _field_prf(self, pred: Sequence[Row], gold: Sequence[Row], field: str) -> PRF:
        """
        Calculate per‑field precision/recall/F1.
        """
        s, g = self._value_set(pred, field), self._value_set(gold, field)
        return PRF(tp=len(s & g), n_pred=len(s), n_gold=len(g))

    def _value_set(self, rows: Sequence[Row], field: str) -> set[tuple[str, str]]:
        """
        Build a set of (document_id, field_value) pairs for a given field.
        Exclude rows where the field value is EMPTY_NA if ignore_empty_na is set to True.
        """
        values = ((r.root_uid, getattr(r, field)) for r in rows)
        return {(doc, v) for doc, v in values if not (self.ignore_empty_na and v == EMPTY_NA)}


def _group_by_doc(rows: Iterable[Row]) -> dict[str, list[Row]]:
    """
    Group rows by their root_uid.
    """
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.root_uid].append(r)
    return groups


###############################################################################
#                                 From files                                  #
###############################################################################

class ScoringInputError(ValueError):
    """
    An exception raised if the gold file has any error, or 
    if the system file cannot be read at all, or 
    if the system file has any error (in strict mode).
    """
    def __init__(self, message: str, issues: Iterable[IngestionIssue]) -> None:
        super().__init__(message)
        self.issues: tuple[IngestionIssue, ...] = tuple(issues)


def _errors(issues: Iterable[IngestionIssue]) -> list[IngestionIssue]:
    return [i for i in issues if i.issue_type == IssueType.ERROR]


def score_claim_frame_files(
    pred_path: Path,
    gold_path: Path,
    scorer: CFEScorer,
    *,
    strict: bool = True,
    root_uids: Iterable[str] | None = None,
) -> tuple[CFEReport, list[IngestionIssue]]:
    """
    Read a system output file and a gold file (12-column output format, or the LDC annotation format
    with extra columns) and score them.

    Args:
        pred_path: System output file.
        gold_path: Gold file.
        scorer:    The CFEScorer used to score the claim frames.
        strict:    If True, any error in the system file raises ScoringInputError. 
                   If False, system rows that could not be read are counted as false positives.
        root_uids: If given, only claim frames of these documents are scored (system and gold alike).
                   Useful when only a subset of the documents is annotated.

    Returns:
        A tuple of (report, issues from both files).

    Raises:
        ScoringInputError: An exception raised if the gold file has any error, or
                           if the system file cannot be read at all,
                           or (strict mode) if the system file has any error.
    """
    gold, gold_issues = read_output_claim_frames(gold_path)
    pred, pred_issues = read_output_claim_frames(pred_path)

    gold_errors = _errors(gold_issues)
    if gold_errors:
        raise ScoringInputError(f"{len(gold_errors)} error(s) in gold file {gold_path}", gold_errors)
    
    pred_errors = _errors(pred_issues)
    row_errors = [i for i in pred_errors if i.location.startswith("line ")]
    if len(row_errors) < len(pred_errors):
        raise ScoringInputError(f"system file {pred_path} cannot be read", pred_errors)
    if row_errors and strict:
        raise ScoringInputError(f"{len(row_errors)} error(s) in system file {pred_path}", row_errors)

    if root_uids is not None:
        wanted = set(root_uids)
        pred = tuple(r for r in pred if r.root_uid.strip() in wanted)
        gold = tuple(r for r in gold if r.root_uid.strip() in wanted)

    report = scorer.score(pred, gold)
    if row_errors:
        # The document of an unreadable row is unknown, so it is counted even when root_uids is given.
        n = len(row_errors)
        claims = report.claims
        report = replace(report, claims=PRF(claims.tp, claims.n_pred + n, claims.n_gold), n_pred_unparsed=n)
    
    return report, [*gold_issues, *pred_issues]