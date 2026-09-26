"""Pick a decision threshold from labeled examples."""

from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..validation import unit_interval, validate_batch


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


@dataclass(frozen=True)
class CalibrationReport:
    recommended: ThresholdPoint
    objective: str
    max_false_positive_rate: Optional[float]
    garbled: int
    clean: int
    points: Tuple[ThresholdPoint, ...]


def _point(
    threshold: float, garbled: Sequence[float], clean: Sequence[float]
) -> ThresholdPoint:
    tp = sum(1 for s in garbled if s >= threshold)
    fp = sum(1 for s in clean if s >= threshold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / len(garbled)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return ThresholdPoint(threshold, precision, recall, f1, fp / len(clean))


def calibrate(
    detector: Any,
    garbled: Iterable[str],
    clean: Iterable[str],
    *,
    objective: str = "f1",
    max_false_positive_rate: Optional[float] = None,
    thresholds: Optional[Iterable[float]] = None,
) -> CalibrationReport:
    """Score both samples once and sweep candidate thresholds.

    objective="f1" picks the highest F1; objective="max_fpr" picks the
    highest recall whose false-positive rate stays within
    max_false_positive_rate, which is required for and only accepted with
    that objective. If no candidate satisfies the limit, the cut 1.0 is
    recommended and recommended.false_positive_rate shows the unmet
    constraint. Ties resolve to the highest threshold. Under
    voting="majority" the ensemble decision counts member votes, so the
    recommended threshold is applied per member; the report still measures
    the aggregate score.

    The recommended threshold is the midpoint of the gap between the chosen
    cut and the highest score below it, so it is not itself an observed
    score; `points` still lists every observed candidate. The midpoint
    applies only when the chosen cut is an observed score and candidates
    were not passed explicitly, which keeps the reported metrics exact; it
    never applies to the 1.0 fallback.
    """
    garbled_texts = list(garbled)
    clean_texts = list(clean)
    if not garbled_texts or not clean_texts:
        raise ValueError(
            "garbled and clean must each contain at least one text"
        )
    validate_batch(garbled_texts)
    validate_batch(clean_texts)
    if objective not in ("f1", "max_fpr"):
        raise ValueError("objective must be 'f1' or 'max_fpr'")
    if objective == "f1" and max_false_positive_rate is not None:
        raise ValueError(
            "max_false_positive_rate requires objective='max_fpr'"
        )
    limit: Optional[float] = None
    if objective == "max_fpr":
        if max_false_positive_rate is None:
            raise ValueError(
                "max_false_positive_rate is required for objective='max_fpr'"
            )
        limit = unit_interval(
            "max_false_positive_rate", max_false_positive_rate
        )
    garbled_scores: List[float] = list(detector.score(garbled_texts))
    clean_scores: List[float] = list(detector.score(clean_texts))
    if thresholds is None:
        candidates = sorted(
            set(garbled_scores) | set(clean_scores) | {0.0, 1.0}
        )
    else:
        candidates = sorted(
            {unit_interval("threshold", t) for t in thresholds}
        )
    points = tuple(_point(t, garbled_scores, clean_scores) for t in candidates)
    fallback = False
    if objective == "f1":
        best = max(points, key=lambda p: (p.f1, p.threshold))
    else:
        assert limit is not None  # set above whenever objective is max_fpr
        eligible = [p for p in points if p.false_positive_rate <= limit]
        if eligible:
            best = max(eligible, key=lambda p: (p.recall, p.threshold))
        else:
            best = _point(1.0, garbled_scores, clean_scores)
            fallback = True
    observed = set(garbled_scores) | set(clean_scores)
    if thresholds is None and not fallback and best.threshold in observed:
        below = [t for t in candidates if t < best.threshold]
        if below:
            # No observed score lies strictly inside the gap, so every
            # metric of the chosen cut holds at the midpoint too.
            midpoint = (max(below) + best.threshold) / 2
            best = replace(best, threshold=midpoint)
    return CalibrationReport(
        best, objective, limit, len(garbled_texts), len(clean_texts), points
    )
