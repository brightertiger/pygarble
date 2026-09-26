"""
Function Word Density Strategy for detecting garbled text.

Natural English text contains ~40-60% function/stop words (the, a, is, etc.).
Garbled text almost never contains these common short words.
"""

import re
from typing import Any, List

from ...data import FUNCTION_WORDS
from ...validation import parameter_value
from ..preprocessing import title_case_ratio
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

    FUNCTION_WORDS = FUNCTION_WORDS

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_ratio: float = parameter_value(
            "min_ratio", kwargs.get("min_ratio", 0.08), 0.08
        )
        self.min_words: int = parameter_value(
            "min_words", kwargs.get("min_words", 5), 5
        )
        self.min_word_length: int = parameter_value(
            "min_word_length", kwargs.get("min_word_length", 2), 2
        )

        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError("min_ratio must be between 0.0 and 1.0")
        if self.min_words < 1:
            raise ValueError("min_words must be at least 1")

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphabetic words; function words are kept at any
        length so "a" and "I" count."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [
            w
            for w in words
            if len(w) >= self.min_word_length or w in self.FUNCTION_WORDS
        ]

    def applicable(self, text: str) -> bool:
        """Abstain on texts with too few analyzable words."""
        return len(self._tokenize(text)) >= self.min_words

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize(text)

        if len(words) < self.min_words:
            return 0.0

        function_count = sum(1 for w in words if w in self.FUNCTION_WORDS)
        ratio = function_count / len(words)

        # Only zero function words across >= 10 words is strong enough
        # evidence to score above 0.5.
        if function_count == 0:
            # Name lists and Title Case headlines legitimately contain
            # zero function words
            if title_case_ratio(text) >= 0.6:
                return 0.3
            if len(words) >= 15:
                return 0.9
            if len(words) >= 10:
                return 0.8
            # 5-9 words: too short to be confident (could be a list
            # of names, technical terms, non-English text, etc.)
            return 0.3

        # At least one function word present: real prose regularly dips
        # below min_ratio (e.g. dense technical sentences), so grade the
        # deficit but never cross the 0.5 decision boundary.
        if ratio < self.min_ratio:
            deficit = (self.min_ratio - ratio) / self.min_ratio
            return min(0.45, 0.2 + deficit * 0.25)

        return 0.0
