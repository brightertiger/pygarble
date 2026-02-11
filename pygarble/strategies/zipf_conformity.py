"""
Zipf's Law Conformity Strategy for detecting garbled text.

Natural language word frequencies follow Zipf's law with heavy repetition.
Garbled text produces flat distributions where every "word" appears once.
Uses Type-Token Ratio and hapax legomena ratio as primary signals.
"""

import re
from collections import Counter
from typing import Any, List

from .base import BaseStrategy


class ZipfConformityStrategy(BaseStrategy):
    """
    Detect garbled text by checking word frequency distribution.

    Natural text has repeated function words giving Type-Token Ratio
    well below 1.0 for 20+ word texts. Garbled text produces unique
    "words" with TTR approaching 1.0.

    Parameters
    ----------
    min_words : int, optional
        Minimum word count for analysis. Default is 20.

    ttr_threshold : float, optional
        TTR above this is suspicious. Default is 0.9.

    hapax_threshold : float, optional
        Hapax legomena ratio above this is suspicious. Default is 0.92.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
    >>> detector.predict("The cat sat on the mat and the dog lay on the rug")
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_words = kwargs.get("min_words", 30)
        self.ttr_threshold = kwargs.get("ttr_threshold", 0.95)
        self.hapax_threshold = kwargs.get("hapax_threshold", 0.95)

        if self.min_words < 5:
            raise ValueError("min_words must be at least 5")
        if not 0.0 <= self.ttr_threshold <= 1.0:
            raise ValueError("ttr_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.hapax_threshold <= 1.0:
            raise ValueError("hapax_threshold must be between 0.0 and 1.0")

    def _tokenize(self, text: str) -> List[str]:
        """Extract lowercase alphabetic words."""
        return re.findall(r"[a-zA-Z]+", text.lower())

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize(text)

        if len(words) < self.min_words:
            return 0.0

        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        ttr = unique_words / total_words

        # Hapax legomena: words appearing exactly once
        hapax_count = sum(
            1 for count in word_counts.values() if count == 1
        )
        hapax_ratio = hapax_count / total_words

        # Perfect uniqueness: every word appears exactly once
        if ttr == 1.0:
            if total_words >= 30:
                return 0.9
            return 0.8

        score = 0.0

        # TTR component
        if ttr > self.ttr_threshold:
            ttr_excess = (ttr - self.ttr_threshold) / (
                1.0 - self.ttr_threshold
            )
            score += ttr_excess * 0.45

        # Hapax ratio component
        if hapax_ratio > self.hapax_threshold:
            hapax_excess = (hapax_ratio - self.hapax_threshold) / (
                1.0 - self.hapax_threshold
            )
            score += hapax_excess * 0.45

        # Combined boost: both signals present
        if ttr > self.ttr_threshold and hapax_ratio > self.hapax_threshold:
            score = min(1.0, score + 0.1)

        return min(1.0, max(0.0, score))

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
