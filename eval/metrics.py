"""Scoring for the detection and robustness harnesses."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

# How a statutory verdict maps to a binary "is this AI?" decision.
#
# The choice is a policy question, not a technical one, so the harness reports
# every policy rather than picking one. `strict` is what a court-facing claim
# would need; `disclosure` is what a platform enforcing a disclosure rule would
# plausibly act on; `permissive` also trusts the local classifier.
VERDICT_POLICIES: dict[str, set[str]] = {
    "strict": {"ai_generated"},
    "disclosure": {"ai_generated", "c2pa_detected_untrusted"},
    "permissive": {"ai_generated", "c2pa_detected_untrusted", "probably_ai_generated"},
}


@dataclass
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        if not (self.precision + self.recall):
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def add(self, predicted_ai: bool, actual_ai: bool) -> None:
        if predicted_ai and actual_ai:
            self.true_positive += 1
        elif predicted_ai and not actual_ai:
            self.false_positive += 1
        elif not predicted_ai and actual_ai:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
        }


def score_policy(rows: Iterable[dict[str, Any]], policy: str) -> ConfusionMatrix:
    """Score prediction rows under one verdict policy."""
    positive_verdicts = VERDICT_POLICIES[policy]
    matrix = ConfusionMatrix()
    for row in rows:
        matrix.add(row["verdict"] in positive_verdicts, row["label"] == "ai")
    return matrix


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — meaningful at the sample sizes we actually have."""
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * ((proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) ** 0.5)
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_matrix(name: str, matrix: ConfusionMatrix) -> str:
    low, high = wilson_interval(matrix.true_positive + matrix.true_negative, matrix.total)
    return (
        f"  {name:<12} n={matrix.total:<4} "
        f"acc={matrix.accuracy:6.1%} [{low:.1%}-{high:.1%}]  "
        f"precision={matrix.precision:6.1%}  recall={matrix.recall:6.1%}  "
        f"specificity={matrix.specificity:6.1%}\n"
        f"               TP={matrix.true_positive} FP={matrix.false_positive} "
        f"TN={matrix.true_negative} FN={matrix.false_negative}"
    )
