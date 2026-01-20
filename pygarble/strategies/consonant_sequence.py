"""
Consonant Sequence Strategy for detecting garbled text.

Detects impossibly long consonant sequences that cannot occur in English.
Conservative threshold to allow valid clusters like 'str', 'scr', 'ngths'.
"""

from .base import BaseStrategy


class ConsonantSequenceStrategy(BaseStrategy):
    """
    Detects garbled text by identifying impossibly long consonant runs.

    English has strict limits on consonant clustering:
    - Word-initial: max ~3 consonants (str-, scr-, spl-)
    - Word-medial: max ~4-5 consonants (strengths has -ngths)
    - Word-final: max ~4 consonants (-ngths, -rlds)

    This strategy uses a conservative threshold of 6+ consonants in a row,
    which is essentially impossible in any English word.
    """

    VOWELS = frozenset("aeiouy")
    CONSONANTS = frozenset("bcdfghjklmnpqrstvwxz")

    def __init__(
        self,
        max_consonants: int = 6,
        min_length: int = 6,
        **kwargs
    ):
        """
        Initialize the consonant sequence strategy.

        Args:
            max_consonants: Max allowed consecutive consonants (default 6)
                           Values 5+ are conservative (allow 'strengths')
            min_length: Minimum text length to analyze (default 6)
        """
        super().__init__(**kwargs)
        self.max_consonants = max_consonants
        self.min_length = min_length

    def _extract_words_for_analysis(self, text: str):
        """Extract words, excluding likely acronyms (all caps)."""
        words = []
        current_word = []
        for c in text:
            if c.isalpha():
                current_word.append(c)
            else:
                if current_word:
                    word = "".join(current_word)
                    # Skip all-uppercase words (likely acronyms)
                    if not word.isupper():
                        words.append(word.lower())
                    current_word = []
        if current_word:
            word = "".join(current_word)
            if not word.isupper():
                words.append(word.lower())
        return " ".join(words)

    def _get_max_consonant_run(self, text: str) -> int:
        """Find the longest consecutive consonant sequence."""
        # Filter out acronyms first
        alpha_text = self._extract_words_for_analysis(text)

        max_run = 0
        current_run = 0

        for c in alpha_text:
            if c in self.CONSONANTS:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return max_run

    def _count_violations(self, text: str) -> tuple:
        """Count consonant sequences exceeding threshold."""
        # Filter out acronyms first
        alpha_text = self._extract_words_for_analysis(text)

        violations = 0
        total_sequences = 0
        current_run = 0

        for c in alpha_text:
            if c in self.CONSONANTS:
                current_run += 1
            else:
                if current_run > 0:
                    total_sequences += 1
                    if current_run > self.max_consonants:
                        violations += 1
                current_run = 0

        # Don't forget the last sequence
        if current_run > 0:
            total_sequences += 1
            if current_run > self.max_consonants:
                violations += 1

        return violations, total_sequences

    def _predict_proba_impl(self, text: str) -> float:
        # Filter out acronyms first
        alpha_text = self._extract_words_for_analysis(text)

        if len(alpha_text) < self.min_length:
            return 0.0

        max_run = self._get_max_consonant_run(text)

        # Immediate flag for extremely long runs
        if max_run >= self.max_consonants + 3:  # 9+ consonants
            return 1.0

        if max_run > self.max_consonants:
            violations, total_seq = self._count_violations(text)
            if total_seq == 0:
                return 0.0

            # Score based on how much over the limit
            excess = max_run - self.max_consonants
            base_score = 0.5 + (excess * 0.15)  # Gradual increase

            # Boost for multiple violations
            if violations > 1:
                base_score = min(1.0, base_score + violations * 0.1)

            return min(1.0, base_score)

        return 0.0

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
