"""Fisher's method over member scores read against a synthetic null."""

import math
from typing import Sequence


def tail_p_value(
    score: float, thresholds: Sequence[float], grid: Sequence[float]
) -> float:
    """Smallest grid probability whose null threshold ``score`` exceeds.

    The walk stops at the first threshold the score does not strictly
    exceed, so a mass point at a threshold (for example 0.0 from a silent
    strategy) never counts as extreme. An empty table gives 1.0.
    """
    p = 1.0
    for threshold, tail in zip(thresholds, grid):
        if not score > threshold:
            break
        p = tail
    return p


def combined_p_value(p_values: Sequence[float]) -> float:
    """Fisher's combined p-value, capped at 1.

    X = -2 * sum(ln p) follows a chi-square law with 2k degrees of freedom
    under independence; its survival function has the closed form
    exp(-X / 2) * sum((X / 2) ** j / j! for j < k). exp(-X / 2) is the
    product of the p-values, which keeps a single member exact.
    """
    half = -sum(math.log(p) for p in p_values)
    term = total = 1.0
    for j in range(1, len(p_values)):
        term *= half / j
        total += term
    return min(1.0, math.prod(p_values) * total)


def fisher_score(p: float, alpha: float) -> float:
    """Map a combined p-value to [0, 1) so that p == alpha scores 0.5."""
    return 1.0 - float(p ** (math.log(0.5) / math.log(alpha)))
