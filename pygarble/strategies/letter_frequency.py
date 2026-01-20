"""
Letter Frequency Strategy for detecting garbled text.

Detects text with letter frequency distributions that deviate extremely
from English norms. Uses a very conservative threshold to avoid false
positives from short text or specialized vocabulary.
"""

from collections import Counter
from .base import BaseStrategy


class LetterFrequencyStrategy(BaseStrategy):
    """
    Detects garbled text by comparing letter frequencies to English norms.

    English has a characteristic letter frequency distribution (ETAOIN...).
    Random gibberish typically deviates significantly from this pattern.

    Uses conservative thresholds to handle:
    - Short text (naturally noisy)
    - Technical jargon
    - Proper nouns and acronyms
    """

    # Expected letter frequencies in English (approximate percentages)
    # From large corpus analysis
    ENGLISH_FREQ = {
        "e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0,
        "n": 6.7, "s": 6.3, "h": 6.1, "r": 6.0, "d": 4.3,
        "l": 4.0, "c": 2.8, "u": 2.8, "m": 2.4, "w": 2.4,
        "f": 2.2, "g": 2.0, "y": 2.0, "p": 1.9, "b": 1.5,
        "v": 1.0, "k": 0.8, "j": 0.15, "x": 0.15, "q": 0.10,
        "z": 0.07,
    }

    # Most common letters - if these are rare, text is suspicious
    TOP_LETTERS = frozenset("etaoinsr")

    # Rare letters - if these dominate, text is suspicious
    RARE_LETTERS = frozenset("jxqz")

    def __init__(
        self,
        deviation_threshold: float = 3.0,
        min_length: int = 20,
        **kwargs
    ):
        """
        Initialize the letter frequency strategy.

        Args:
            deviation_threshold: Standard deviations from norm (default 3.0)
                                Higher = more conservative (fewer false positives)
            min_length: Minimum text length to analyze (default 20)
        """
        super().__init__(**kwargs)
        self.deviation_threshold = deviation_threshold
        self.min_length = min_length

    def _calculate_chi_squared(self, text: str) -> float:
        """Calculate chi-squared statistic for letter frequencies."""
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0

        observed = Counter(alpha_text)
        total = len(alpha_text)

        chi_squared = 0.0
        for letter, expected_pct in self.ENGLISH_FREQ.items():
            observed_count = observed.get(letter, 0)
            expected_count = (expected_pct / 100) * total

            # Only include if expected count is meaningful
            if expected_count >= 1:
                diff = observed_count - expected_count
                chi_squared += (diff ** 2) / expected_count

        return chi_squared

    def _check_extreme_patterns(self, text: str) -> float:
        """Check for extremely abnormal letter patterns."""
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0

        counts = Counter(alpha_text)
        total = len(alpha_text)

        # Check if rare letters dominate
        rare_count = sum(counts.get(c, 0) for c in self.RARE_LETTERS)
        rare_ratio = rare_count / total

        # Check if common letters are absent
        common_count = sum(counts.get(c, 0) for c in self.TOP_LETTERS)
        common_ratio = common_count / total

        score = 0.0

        # Rare letters normally make up ~0.5% of text
        # If > 15%, very suspicious
        if rare_ratio > 0.15:
            score += 0.4

        # Common letters normally make up ~60% of text
        # If < 30%, very suspicious
        if common_ratio < 0.30:
            score += 0.4

        # Check for single letter dominance
        if counts:
            max_freq = max(counts.values()) / total
            # Any single letter > 30% is suspicious (normal max is ~13%)
            if max_freq > 0.30:
                score += 0.3

        return min(1.0, score)

    def _predict_proba_impl(self, text: str) -> float:
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0

        # Check extreme patterns first (faster)
        extreme_score = self._check_extreme_patterns(text)
        if extreme_score >= 0.7:
            return extreme_score

        # Calculate chi-squared statistic
        chi_sq = self._calculate_chi_squared(text)

        # Normalize by degrees of freedom (26-1 = 25 for letters)
        # and text length factor
        normalized = chi_sq / (25 * (1 + len(alpha_text) / 100))

        # Very conservative threshold
        if normalized > self.deviation_threshold:
            return min(1.0, 0.4 + (normalized - self.deviation_threshold) * 0.1)

        return max(extreme_score, normalized / self.deviation_threshold * 0.3)

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
