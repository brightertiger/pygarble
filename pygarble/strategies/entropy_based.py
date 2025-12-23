import math
from collections import Counter
from typing import Set

from .base import BaseStrategy


COMMON_BIGRAMS: Set[str] = {
    "th", "he", "in", "en", "nt", "re", "er", "an", "ti", "es",
    "on", "at", "se", "nd", "or", "ar", "al", "te", "co", "de",
    "to", "ra", "et", "ed", "it", "sa", "em", "ro", "of", "is",
    "ou", "le", "ve", "ng", "ha", "as", "ma", "ll", "io", "ea",
}


class EntropyBasedStrategy(BaseStrategy):
    def _calculate_entropy(self, char_counts: Counter) -> float:
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy

    def _get_bigrams(self, text: str) -> list:
        alpha_text = "".join(c.lower() for c in text if c.isalpha())
        return [alpha_text[i:i+2] for i in range(len(alpha_text) - 1)]

    def _get_common_bigram_ratio(self, text: str) -> float:
        bigrams = self._get_bigrams(text)
        if not bigrams:
            return 0.0
        common_count = sum(1 for bg in bigrams if bg in COMMON_BIGRAMS)
        return common_count / len(bigrams)

    def _predict_impl(self, text: str) -> bool:
        char_counts = self._get_alpha_char_counts(text)
        if not char_counts:
            return False

        entropy = self._calculate_entropy(char_counts)
        entropy_threshold = self.kwargs.get("entropy_threshold", 2.5)

        if entropy < entropy_threshold:
            return True

        bigram_ratio = self._get_common_bigram_ratio(text)
        bigram_threshold = self.kwargs.get("bigram_threshold", 0.15)

        if bigram_ratio < bigram_threshold:
            return True

        return False

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_alpha_char_counts(text)
        if not char_counts:
            return 0.0

        entropy = self._calculate_entropy(char_counts)
        max_entropy = math.log2(len(char_counts)) if char_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_score = 1.0 - normalized_entropy

        bigram_ratio = self._get_common_bigram_ratio(text)
        bigram_score = 1.0 - min(bigram_ratio / 0.3, 1.0)

        return max(entropy_score * 0.4 + bigram_score * 0.6, min(entropy_score, bigram_score))
