"""
Word lookup strategy for garble detection.

Uses an embedded set of common English words to validate text.
No external dependencies required.
"""

import re
from typing import Any, List

from .base import BaseStrategy
from ..data import ENGLISH_WORDS


class WordLookupStrategy(BaseStrategy):
    """
    Detect garbled text by checking words against a dictionary.

    This strategy tokenizes input text and checks what proportion
    of words appear in the embedded English word set (50,000 words
    derived from Peter Norvig's word frequency list).

    This is a dependency-free alternative to EnglishWordValidationStrategy
    which requires pyspellchecker.

    Parameters
    ----------
    unknown_threshold : float, optional
        Proportion of unknown words above which text is considered
        garbled. Default is 0.5 (50% unknown words).

    min_word_length : int, optional
        Minimum word length to check. Shorter words are ignored.
        Default is 2.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.WORD_LOOKUP)
    >>> detector.predict("hello world")
    False
    >>> detector.predict("xyzzy plugh")
    True
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.unknown_threshold = kwargs.get("unknown_threshold", 0.5)
        self.min_word_length = kwargs.get("min_word_length", 2)

        if not 0.0 <= self.unknown_threshold <= 1.0:
            raise ValueError("unknown_threshold must be between 0.0 and 1.0")
        if self.min_word_length < 1:
            raise ValueError("min_word_length must be at least 1")

    def _tokenize(self, text: str) -> List[str]:
        """Extract words from text."""
        # Find all alphabetic word sequences
        words = re.findall(r"[a-zA-Z]+", text.lower())
        # Filter by minimum length
        return [w for w in words if len(w) >= self.min_word_length]

    def _compute_unknown_ratio(self, text: str) -> float:
        """
        Compute the ratio of words not found in the dictionary.

        Returns a value between 0 and 1 where higher means more
        unknown words (more likely garbled).
        """
        words = self._tokenize(text)

        if not words:
            return 0.0  # No words to check

        unknown_count = sum(1 for w in words if w not in ENGLISH_WORDS)
        return unknown_count / len(words)

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on word lookup.

        Returns a value between 0 and 1 where:
        - 0.0 = all words found in dictionary
        - 1.0 = all words unknown

        The raw unknown ratio is used directly as the garble score,
        as it naturally maps to the probability of being garbled.
        """
        return self._compute_unknown_ratio(text)
