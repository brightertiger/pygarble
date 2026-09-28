"""Bigram likelihood of the text against shuffles of its own letters."""

from typing import Any

from ...validation import positive_int
from ..measures import permutation_gap
from ._windowed import WindowedStrategy


class PermutationTestStrategy(WindowedStrategy):
    """How much likelier the text is than shuffles of its own letters.

    English letter order is far more probable under an English bigram
    model than the same letters shuffled. It detects text whose letter
    order carries no English structure, such as keyboard mash or random
    letters. Pronounceable invented words already have English-like
    letter order, so it often misses them at any length; it is the
    weakest of the windowed strategies on such text. Shuffles are
    deterministic. This is an English-reference method: other languages
    written in Latin letters may be flagged, more so the less they
    resemble English, and text with no ASCII letters is not scored. It
    needs at least ``min_length`` letters (default 8).

    Args:
        midpoint: standardised value at which the score is 0.5
            (default 1.5; 1.0 is the null's 99th percentile)
        scale: sigmoid steepness, positive (default 2.0)
        min_length: minimum normalised characters to judge (default 8)
        shuffles: shuffles averaged per window (default 8)

    Example:
        >>> detector = GarbleDetector(Strategy.PERMUTATION_TEST)
        >>> detector.predict("The meeting has been moved to Thursday.")
        False
    """

    statistic = "permutation_test"
    reason = "permutation_gap"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shuffles = positive_int("shuffles", kwargs.get("shuffles", 8))

    def _raw(self, window: str) -> float:
        from ...data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB

        return permutation_gap(
            window, BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB, self.shuffles
        )
