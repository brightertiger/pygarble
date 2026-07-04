"""
Affix Detection Strategy for detecting garbled text.

English words contain recognizable morphemes: prefixes (un-, re-, pre-)
and suffixes (-tion, -ing, -ly, -ness). Garbled text rarely produces
words with these patterns.
"""

import re
from typing import Any, List

from .base import BaseStrategy


class AffixDetectionStrategy(BaseStrategy):
    """
    Detect garbled text by checking for recognizable English affixes.

    English words frequently contain common prefixes and suffixes.
    Text composed of random character sequences rarely produces words
    with recognizable morphological structure.

    This is intentionally a weak signal - many legitimate text types
    (names, technical terms) may lack common affixes.

    Parameters
    ----------
    min_affix_ratio : float, optional
        Minimum ratio of words with affixes. Default is 0.2.

    min_word_length : int, optional
        Minimum word length to analyze. Default is 4.

    min_analyzable_words : int, optional
        Minimum analyzable words required. Default is 5.

    min_stem_length : int, optional
        Minimum remaining stem after affix removal. Default is 2.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.AFFIX_DETECTION)
    >>> detector.predict("The programming language is incredibly powerful")
    False
    """

    PREFIXES = (
        "un", "re", "pre", "dis", "mis", "over", "under", "out",
        "sub", "super", "inter", "trans", "non", "anti", "auto",
        "semi", "multi", "counter", "extra", "ultra",
    )

    SUFFIXES = (
        "tion", "sion", "ment", "ness", "able", "ible",
        "ous", "ious", "ive", "ful", "less",
        "ing", "ting", "ling",
        "ized", "ised", "ize", "ise",
        "ify",
        "ated", "ator",
        "ally", "ially", "ably", "ibly",
        "ence", "ance", "ency", "ancy",
        "ical", "ular",
        "ly", "er", "est", "ed", "en",
        "al", "ial", "ary", "ory",
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_affix_ratio = kwargs.get("min_affix_ratio", 0.2)
        self.min_word_length = kwargs.get("min_word_length", 4)
        self.min_analyzable_words = kwargs.get("min_analyzable_words", 5)
        self.min_stem_length = kwargs.get("min_stem_length", 2)

        if not 0.0 <= self.min_affix_ratio <= 1.0:
            raise ValueError("min_affix_ratio must be between 0.0 and 1.0")
        if self.min_word_length < 2:
            raise ValueError("min_word_length must be at least 2")

    def _tokenize(self, text: str) -> List[str]:
        """Extract lowercase alphabetic words meeting minimum length."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in words if len(w) >= self.min_word_length]

    @staticmethod
    def _plausible_stem(stem: str) -> bool:
        """A real English stem contains at least one vowel; random
        consonant strings that happen to end in 'er'/'ed' do not."""
        return any(c in "aeiouy" for c in stem)

    def _has_prefix(self, word: str) -> bool:
        """Check if word starts with a known prefix with sufficient stem."""
        for prefix in self.PREFIXES:
            if word.startswith(prefix) and len(word) - len(prefix) >= self.min_stem_length:
                if self._plausible_stem(word[len(prefix):]):
                    return True
        return False

    def _has_suffix(self, word: str) -> bool:
        """Check if word ends with a known suffix with sufficient stem."""
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= self.min_stem_length:
                if self._plausible_stem(word[:-len(suffix)]):
                    return True
        return False

    def _has_affix(self, word: str) -> bool:
        """Check if word contains any recognizable affix."""
        return self._has_prefix(word) or self._has_suffix(word)

    def applicable(self, text: str) -> bool:
        """Affix statistics need a minimum number of analyzable words."""
        return len(self._tokenize(text)) >= self.min_analyzable_words

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize(text)

        if len(words) < self.min_analyzable_words:
            return 0.0

        affix_count = sum(1 for w in words if self._has_affix(w))
        ratio = affix_count / len(words)

        if ratio >= self.min_affix_ratio:
            return 0.0

        # Zero affixes is a weak signal by itself: legitimate name lists
        # and technical jargon may lack affixes entirely, so it must not
        # cross the 0.5 decision threshold without volume behind it.
        if affix_count == 0:
            if len(words) >= 20:
                return 0.5
            if len(words) >= 10:
                return 0.45
            # 5-9 words, zero affixes: mild signal
            return 0.35

        # Some affixes but below the minimum ratio: graded score that
        # grows with the deficit and with the amount of evidence
        deficit = (self.min_affix_ratio - ratio) / self.min_affix_ratio
        score = 0.3 + deficit * 0.25
        if len(words) >= 20:
            score += 0.15
        return min(0.75, score)
