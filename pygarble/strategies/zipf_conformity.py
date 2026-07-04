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
from .function_word_density import FunctionWordDensityStrategy
from ..data import ENGLISH_WORDS


class ZipfConformityStrategy(BaseStrategy):
    """
    Detect garbled text by checking word frequency distribution.

    Natural text has repeated function words giving Type-Token Ratio
    well below 1.0 for 30+ word texts. Garbled text produces unique
    "words" with TTR approaching 1.0.

    A flat distribution alone is NOT proof of gibberish (a list of 30
    distinct real words also has TTR 1.0), so scores above 0.5 require
    corroboration: the text must also contain zero function words and
    a majority of words unknown to the embedded dictionary. Without
    corroboration the score is capped below 0.5.

    Parameters
    ----------
    min_words : int, optional
        Minimum word count for analysis. Default is 30.

    ttr_threshold : float, optional
        TTR above this is suspicious. Default is 0.95.

    hapax_threshold : float, optional
        Hapax legomena ratio above this is suspicious. Default is 0.95.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.ZIPF_CONFORMITY)
    >>> detector.predict("The cat sat on the mat and the dog lay on the rug")
    False
    """

    # Score cap applied when a flat distribution lacks corroborating
    # evidence of gibberish (kept below the 0.5 decision boundary).
    UNCORROBORATED_CAP = 0.45

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

    def applicable(self, text: str) -> bool:
        """Abstain on texts with too few words for distribution stats."""
        return len(self._tokenize(text)) >= self.min_words

    @staticmethod
    def _corroborated(words: List[str]) -> bool:
        """
        Check for independent evidence that a flat word distribution is
        actually gibberish rather than a legitimate list of distinct
        real words (names, ingredients, keywords, ...).
        """
        if any(
            w in FunctionWordDensityStrategy.FUNCTION_WORDS for w in words
        ):
            return False
        unknown = sum(1 for w in words if w not in ENGLISH_WORDS)
        return unknown / len(words) >= 0.5

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
            score = 0.9
        else:
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

        score = min(1.0, max(0.0, score))

        # A flat distribution by itself is weak evidence: cap below the
        # decision boundary unless corroborated by zero function words
        # and mostly-unknown vocabulary.
        if score > self.UNCORROBORATED_CAP and not self._corroborated(words):
            return self.UNCORROBORATED_CAP

        return score
