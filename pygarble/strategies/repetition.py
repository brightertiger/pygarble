"""
Repetition detection strategy for garble detection.

Detects text with excessive character or pattern repetition.
"""

import re
from typing import Any

from .base import BaseStrategy


class RepetitionStrategy(BaseStrategy):
    """
    Detect garbled text based on character/pattern repetition.

    This strategy identifies:
    - Repeated single characters (aaaaaaa)
    - Repeated bigrams (ababababab)
    - Repeated trigrams (abcabcabc)
    - Repeated words (test test test)
    - Low character diversity

    Parameters
    ----------
    max_char_repeat : int, optional
        Maximum allowed consecutive repeated characters. Default is 3.

    max_pattern_repeat : int, optional
        Maximum allowed pattern repetitions. Default is 3.

    diversity_threshold : float, optional
        Minimum ratio of unique characters to total. Default is 0.3.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.REPETITION)
    >>> detector.predict("aaaaaaaaaa")
    True
    >>> detector.predict("abababababab")
    True
    >>> detector.predict("hello world")
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_char_repeat = kwargs.get("max_char_repeat", 3)
        self.max_pattern_repeat = kwargs.get("max_pattern_repeat", 3)
        self.diversity_threshold = kwargs.get("diversity_threshold", 0.3)

        if self.max_char_repeat < 1:
            raise ValueError("max_char_repeat must be at least 1")
        if self.max_pattern_repeat < 1:
            raise ValueError("max_pattern_repeat must be at least 1")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be between 0.0 and 1.0")

        # Compile patterns for efficiency.
        # Only alphanumeric characters count as character repetition:
        # whitespace runs and formatting characters (----, ====, ....) are
        # normal in real documents.
        self._repeated_char_pattern = re.compile(r"([a-z0-9])\1{" + str(self.max_char_repeat) + r",}")
        self._repeated_bigram_pattern = re.compile(r"(.{2})\1{" + str(self.max_pattern_repeat) + r",}")
        self._repeated_trigram_pattern = re.compile(r"(.{3})\1{" + str(self.max_pattern_repeat - 1) + r",}")
        self._word_pattern = re.compile(r"[a-z0-9]+")

    def _check_char_repetition(self, text: str) -> float:
        """Check for repeated single characters."""
        matches = self._repeated_char_pattern.findall(text.lower())
        if matches:
            # Find longest repetition
            all_matches = self._repeated_char_pattern.finditer(text.lower())
            max_len = max(len(m.group(0)) for m in all_matches)
            # Score based on length of repetition
            return min(1.0, max_len / 10.0)
        return 0.0

    def _check_pattern_repetition(self, text: str) -> float:
        """Check for repeated bigram/trigram patterns."""
        text_lower = text.lower()
        score = 0.0

        # Check bigram repetition (repeating unit must contain something
        # alphanumeric; repeated whitespace/formatting is not garble)
        bigram_matches = [
            m for m in self._repeated_bigram_pattern.finditer(text_lower)
            if any(ch.isalnum() for ch in m.group(1))
        ]
        if bigram_matches:
            max_len = max(len(m.group(0)) for m in bigram_matches)
            score = max(score, min(1.0, max_len / 12.0))

        # Check trigram repetition
        trigram_matches = [
            m for m in self._repeated_trigram_pattern.finditer(text_lower)
            if any(ch.isalnum() for ch in m.group(1))
        ]
        if trigram_matches:
            max_len = max(len(m.group(0)) for m in trigram_matches)
            score = max(score, min(1.0, max_len / 15.0))

        return score

    def _check_word_repetition(self, text: str) -> float:
        """
        Check for repeated words (test test test) or repeated
        two-word cycles (foo bar foo bar foo bar).
        """
        words = self._word_pattern.findall(text.lower())
        if len(words) < 3:
            return 0.0

        # Single repeated token dominating the text
        top_count = max(words.count(w) for w in set(words))
        top_ratio = top_count / len(words)
        if top_count >= 3 and top_ratio >= 0.6:
            return min(1.0, top_ratio)

        # Repeated two-word cycle (needs at least 2 full cycles + 1 word)
        if len(words) >= 5 and words[0] != words[1]:
            if all(w == words[i % 2] for i, w in enumerate(words)):
                return 0.8

        return 0.0

    def _check_diversity(self, text: str) -> float:
        """
        Check character diversity.

        Low diversity (few unique characters) suggests garbled text.
        """
        # Only consider alphanumeric for diversity
        chars = [c.lower() for c in text if c.isalnum()]

        if len(chars) < 4:
            return 0.0

        unique_chars = len(set(chars))

        # There are only ~36 possible alphanumeric characters, so raw
        # unique/total shrinks with length for perfectly normal text.
        # Compare against the best diversity achievable at this length
        # so long paragraphs are judged fairly.
        max_possible = min(len(chars), 36)
        diversity = unique_chars / max_possible

        if diversity >= self.diversity_threshold:
            return 0.0
        else:
            # Low diversity = high garble score
            return 1.0 - (diversity / self.diversity_threshold)

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on repetition analysis.

        Combines multiple repetition signals.
        """
        if not text or len(text) < 3:
            return 0.0

        # Calculate individual scores
        char_rep_score = self._check_char_repetition(text)
        pattern_rep_score = self._check_pattern_repetition(text)
        diversity_score = self._check_diversity(text)
        word_rep_score = self._check_word_repetition(text)

        # Combine scores - take maximum (any repetition type is suspicious)
        combined_score = max(
            char_rep_score, pattern_rep_score, diversity_score, word_rep_score
        )

        return min(1.0, combined_score)
