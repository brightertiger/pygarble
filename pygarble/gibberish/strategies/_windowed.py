"""Shared scoring for the windowed English-reference strategies."""

import statistics
from abc import abstractmethod
from functools import lru_cache
from typing import Any

from ...validation import finite_number, positive_int
from ..analysis import Evidence
from ..measures import bucket, standardised, windows
from ..preprocessing import TextFeatures
from ..scoring import sigmoid
from .base import BaseStrategy


@lru_cache(maxsize=None)
def reference_text() -> str:
    """Reference words joined with the most frequent last."""
    from ...data import REFERENCE_WORDS

    return " ".join(reversed(REFERENCE_WORDS))


class WindowedStrategy(BaseStrategy):
    """Score a raw statistic against the synthetic English null.

    The normalised text (lowercase ASCII words, structured and allowlisted
    tokens removed) is split into windows of at most 127 characters. Each
    window's statistic is standardised against the null for its length
    bucket, and the median over windows is mapped through a sigmoid.
    """

    # Key into STATISTIC_NULL and the reason reported with the score.
    statistic: str
    reason: str

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.midpoint = finite_number("midpoint", kwargs.get("midpoint", 1.5))
        self.scale = finite_number("scale", kwargs.get("scale", 2.0))
        self.min_length = positive_int(
            "min_length", kwargs.get("min_length", 8)
        )
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    @abstractmethod
    def _raw(self, window: str) -> float:
        """The statistic on one window; higher is less like English."""

    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return len(" ".join(TextFeatures(text).ascii_words)) >= self.min_length

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        from ...data import STATISTIC_NULL

        text = " ".join(features.ascii_words)
        if len(text) < self.min_length:
            return Evidence(0.0, False, "insufficient_text")
        table = STATISTIC_NULL[self.statistic]
        z = statistics.median(
            standardised(self._raw(window), bucket(len(window)), table)
            for window in windows(text)
        )
        return Evidence(
            sigmoid(self.scale * (z - self.midpoint)), True, self.reason
        )

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
