"""
N-gram frequency strategy for garble detection.

Analyzes the proportion of character trigrams that appear in
common English text. Text with many uncommon trigrams is flagged.
"""

from typing import Any

from .base import BaseStrategy
from ..data import COMMON_TRIGRAMS


class NGramFrequencyStrategy(BaseStrategy):
    """
    Detect garbled text using character trigram frequency analysis.

    This strategy checks what proportion of trigrams in the input
    text appear in the set of common English trigrams. Text with
    few common trigrams is flagged as garbled.

    Parameters
    ----------
    common_ratio_threshold : float, optional
        Minimum ratio of common trigrams required for text to be
        considered valid. Default is 0.3 (30% of trigrams must be
        in the common set).

    min_length : int, optional
        Minimum text length to analyze. Shorter texts return 0.0.
        Default is 4 (minimum to have one trigram).

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
    >>> detector.predict("the quick brown fox")
    False
    >>> detector.predict("xzqwkjhf")
    True
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.common_ratio_threshold = kwargs.get("common_ratio_threshold", 0.3)
        self.min_length = kwargs.get("min_length", 4)

        if not 0.0 <= self.common_ratio_threshold <= 1.0:
            raise ValueError("common_ratio_threshold must be between 0.0 and 1.0")
        if self.min_length < 1:
            raise ValueError("min_length must be at least 1")

    def _extract_trigrams(self, text: str) -> list:
        """Extract all alphabetic trigrams from text."""
        text = text.lower()
        trigrams = []

        # Extract trigrams from each word
        words = text.split()
        for word in words:
            # Keep only alphabetic characters
            word = "".join(c for c in word if c.isalpha())
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigrams.append(word[i:i + 3])

        return trigrams

    def _compute_common_ratio(self, text: str) -> float:
        """
        Compute the ratio of trigrams that are common in English.

        Returns a value between 0 and 1, where higher means more
        English-like (more common trigrams).
        """
        trigrams = self._extract_trigrams(text)

        if not trigrams:
            return 1.0  # No trigrams = can't determine, assume valid

        common_count = sum(1 for t in trigrams if t in COMMON_TRIGRAMS)
        return common_count / len(trigrams)

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on trigram analysis.

        Returns a value between 0 and 1 where:
        - 0.0 = definitely valid English (many common trigrams)
        - 1.0 = definitely garbled (few common trigrams)
        """
        # Check minimum length
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < self.min_length:
            return 0.0

        common_ratio = self._compute_common_ratio(text)

        # Map common ratio to garble score
        # High common_ratio = low garble score
        # Low common_ratio = high garble score

        if common_ratio >= self.common_ratio_threshold:
            # Above threshold: scale 0.0 to 0.4
            # common_ratio 1.0 -> 0.0, common_ratio threshold -> 0.4
            range_above = 1.0 - self.common_ratio_threshold
            if range_above > 0:
                normalized = (common_ratio - self.common_ratio_threshold) / range_above
                garble_score = 0.4 * (1.0 - normalized)
            else:
                garble_score = 0.0
        else:
            # Below threshold: scale 0.5 to 1.0
            # common_ratio threshold -> 0.5, common_ratio 0.0 -> 1.0
            if self.common_ratio_threshold > 0:
                normalized = common_ratio / self.common_ratio_threshold
                garble_score = 1.0 - (0.5 * normalized)
            else:
                garble_score = 1.0

        return min(1.0, max(0.0, garble_score))
