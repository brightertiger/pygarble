"""Cavnar-Trenkle rank distance from an English character n-gram profile."""

from functools import lru_cache
from typing import Dict

from ..measures import ngram_rank_distance
from ._windowed import WindowedStrategy


@lru_cache(maxsize=None)
def _ranks() -> Dict[str, int]:
    from ...data import NGRAM_RANKS

    return {gram: rank for rank, gram in enumerate(NGRAM_RANKS)}


class NGramRankStrategy(WindowedStrategy):
    """Out-of-place distance from English character n-gram ranks.

    The text's 1- to 3-grams are ranked by frequency and compared with the
    ranks of the same n-grams in English; n-grams English rarely uses
    carry the largest penalty. This is an English-reference method: other
    languages written in Latin letters may be flagged, more so the less
    they resemble English, and text with no ASCII letters is not scored.
    It needs at least ``min_length`` letters (default 8) and grows more
    reliable with length.

    Args:
        midpoint: standardised value at which the score is 0.5
            (default 1.5; 1.0 is the null's 99th percentile)
        scale: sigmoid steepness, positive (default 2.0)
        min_length: minimum normalised characters to judge (default 8)

    Example:
        >>> detector = GarbleDetector(Strategy.NGRAM_RANK)
        >>> detector.predict("The meeting has been moved to Thursday.")
        False
    """

    statistic = "ngram_rank"
    reason = "ngram_rank_distance"

    def _raw(self, window: str) -> float:
        return ngram_rank_distance(window, _ranks())
