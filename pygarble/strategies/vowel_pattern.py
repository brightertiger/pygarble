"""
Vowel Pattern Strategy for detecting garbled text.

Detects invalid vowel sequences and patterns that don't occur in English.
Conservative approach to handle legitimate words with unusual vowel patterns.
"""

from .base import BaseStrategy


class VowelPatternStrategy(BaseStrategy):
    """
    Detects garbled text by analyzing vowel patterns.

    English has constraints on vowel sequences:
    - Very few words have 3+ consecutive vowels
    - Valid: 'queue', 'beautiful', 'onomatopoeia'
    - Most 4+ vowel sequences are garbled

    Uses conservative thresholds to avoid false positives.
    """

    VOWELS = frozenset("aeiou")  # Not including 'y' for stricter vowel check

    # Valid long vowel sequences found in real English words
    VALID_LONG_VOWELS = frozenset({
        "eau",   # beautiful, bureau
        "iou",   # precious, cautious
        "uou",   # continuous, strenuous
        "eou",   # gorgeous, courteous
        "aeo",   # onomatopoeia
        "oeia",  # onomatopoeia
        "oeio",  # part of onomatopoeia
        "ueue",  # queue
        "uee",   # queen (not really 3 vowels but included for safety)
        "ooe",   # wooed
        "aie",   # gaiety
        "oui",   # Louis, Louisiana
        "uai",   # quail-like patterns
        "eai",   # reality (in some accents)
        "eio",   # ratio, patio-like
    })

    def __init__(
        self,
        max_vowel_run: int = 4,
        min_length: int = 5,
        **kwargs
    ):
        """
        Initialize the vowel pattern strategy.

        Args:
            max_vowel_run: Max consecutive vowels before flagging (default 4)
            min_length: Minimum text length to analyze (default 5)
        """
        super().__init__(**kwargs)
        self.max_vowel_run = max_vowel_run
        self.min_length = min_length

    def _get_vowel_sequences(self, text: str):
        """Extract all vowel sequences from text."""
        alpha_text = "".join(c.lower() for c in text if c.isalpha())
        sequences = []
        current_seq = []

        for c in alpha_text:
            if c in self.VOWELS:
                current_seq.append(c)
            else:
                if len(current_seq) >= 3:  # Only interested in 3+
                    sequences.append("".join(current_seq))
                current_seq = []

        if len(current_seq) >= 3:
            sequences.append("".join(current_seq))

        return sequences

    def _is_valid_vowel_sequence(self, seq: str) -> bool:
        """Check if a vowel sequence is valid in English."""
        # Check against known valid patterns
        for valid in self.VALID_LONG_VOWELS:
            if valid in seq or seq in valid:
                return True
        return False

    def _predict_proba_impl(self, text: str) -> float:
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0

        sequences = self._get_vowel_sequences(text)

        if not sequences:
            return 0.0

        invalid_count = 0
        max_invalid_length = 0

        for seq in sequences:
            # Very long sequences are almost always invalid
            if len(seq) > self.max_vowel_run:
                if not self._is_valid_vowel_sequence(seq):
                    invalid_count += 1
                    max_invalid_length = max(max_invalid_length, len(seq))

        if invalid_count == 0:
            return 0.0

        # Score based on length and count of invalid sequences
        base_score = 0.4 + (max_invalid_length - self.max_vowel_run) * 0.1
        base_score += (invalid_count - 1) * 0.15

        return min(1.0, base_score)

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
