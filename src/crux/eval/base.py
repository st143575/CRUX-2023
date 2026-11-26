"""
Building blocks shared by the Task 1 and Task 2 scorers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class Scorer[T, R](Protocol):
    """
    Protocol for a scorer that evaluates system items against gold items.

    Configuration (weights, options, etc.) lives in the scorer object;
    the data to score is passed to score(), so one scorer can score multiple items.
    """
    def score(self, pred: Sequence[T], gold: Sequence[T], /) -> R: ...


@dataclass(frozen=True)
class PRF:
    """
    Precision, recall and F1 computed from raw counts. A 0/0 ratio is scored as 0.

    Attributes:
        tp: Number of true positives.
        n_pred: Number of system items (TP + FP).
        n_gold: Number of gold items (TP + FN).
    """
    tp: int
    n_pred: int
    n_gold: int

    def __post_init__(self):
        if self.tp < 0 or self.n_pred < 0 or self.n_gold < 0:
            raise ValueError(f"counts must be non-negative: {self}")
        if self.tp > self.n_pred or self.tp > self.n_gold:
            raise ValueError(f"tp cannot exceed n_pred or n_gold: {self}")

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp
    
    @property
    def fn(self) -> int:
        return self.n_gold - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            # "n_pred": self.n_pred, "n_gold": self.n_gold,
        }