"""
Function Word Density Strategy for detecting garbled text.

Natural English text contains ~40-60% function/stop words (the, a, is, etc.).
Garbled text almost never contains these common short words.
"""

import re
from typing import Any, List

from .base import BaseStrategy


class FunctionWordDensityStrategy(BaseStrategy):
    """
    Detect garbled text by checking for common English function words.

    English text naturally includes articles, prepositions, pronouns,
    and conjunctions. Random garbled text almost never produces these.

    Parameters
    ----------
    min_ratio : float, optional
        Minimum expected function word ratio. Default is 0.08.

    min_words : int, optional
        Minimum analyzable words required. Default is 5.

    min_word_length : int, optional
        Minimum word length to count. Default is 2.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
    >>> detector.predict("The cat sat on a mat")
    False
    >>> detector.predict("xkrf plmq bvzt nwsd jghc trbn mkpl wqzd lpnr fvxt")
    True
    """

    FUNCTION_WORDS = frozenset({
        # Articles
        "the", "a", "an",
        # Prepositions
        "of", "in", "to", "for", "on", "at", "by", "from", "with",
        "up", "out", "about", "into", "over", "after", "as",
        # Conjunctions
        "and", "but", "or", "nor", "so", "yet", "if", "then",
        "than", "that", "when", "while", "because", "although",
        # Pronouns
        "i", "me", "my", "we", "us", "our", "you", "your",
        "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "this", "these", "those",
        # Auxiliary/common verbs
        "is", "am", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "shall", "should",
        "may", "might", "must",
        # Other high-frequency words
        "not", "no", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such",
        "what", "which", "who", "how", "where",
        "very", "just", "also", "too",
    })

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_ratio = kwargs.get("min_ratio", 0.08)
        self.min_words = kwargs.get("min_words", 5)
        self.min_word_length = kwargs.get("min_word_length", 2)

        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError("min_ratio must be between 0.0 and 1.0")
        if self.min_words < 1:
            raise ValueError("min_words must be at least 1")

    def _tokenize(self, text: str) -> List[str]:
        """Extract lowercase alphabetic words."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in words if len(w) >= self.min_word_length]

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize(text)

        if len(words) < self.min_words:
            return 0.0

        function_count = sum(1 for w in words if w in self.FUNCTION_WORDS)
        ratio = function_count / len(words)

        # Strong signal: many words and zero function words
        if function_count == 0:
            if len(words) >= 15:
                return 0.9
            if len(words) >= 10:
                return 0.8
            # 5-9 words: too short to be confident (could be a list
            # of names, technical terms, non-English text, etc.)
            return 0.3

        # Below threshold: scale score based on word count
        if ratio < self.min_ratio:
            deficit = (self.min_ratio - ratio) / self.min_ratio
            if len(words) >= 10:
                return min(0.8, 0.5 + deficit * 0.2)
            return min(0.4, 0.2 + deficit * 0.2)

        return 0.0

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
