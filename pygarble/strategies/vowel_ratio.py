import re
from typing import List

from .base import BaseStrategy


VOWELS = frozenset("aeiou")
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")


class VowelRatioStrategy(BaseStrategy):
    def _get_vowel_ratio(self, text: str) -> float:
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if not alpha_chars:
            return 0.0

        vowel_count = sum(1 for c in alpha_chars if c in VOWELS)
        return vowel_count / len(alpha_chars)

    def _has_consonant_cluster(self, text: str) -> bool:
        cluster_len = self.kwargs.get("consonant_cluster_len", 4)
        pattern = rf"[bcdfghjklmnpqrstvwxyz]{{{cluster_len},}}"
        return bool(re.search(pattern, text.lower()))

    def _get_max_consonant_run(self, text: str) -> int:
        max_run = 0
        current_run = 0
        for c in text.lower():
            if c in CONSONANTS:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def _predict_impl(self, text: str) -> bool:
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if not alpha_chars:
            return False

        ratio = self._get_vowel_ratio(text)
        min_ratio = self.kwargs.get("min_vowel_ratio", 0.15)
        max_ratio = self.kwargs.get("max_vowel_ratio", 0.65)

        if ratio < min_ratio or ratio > max_ratio:
            return True

        max_consonant_run = self._get_max_consonant_run(text)
        if max_consonant_run >= self.kwargs.get("consonant_cluster_len", 4):
            return True

        return False

    def _predict_proba_impl(self, text: str) -> float:
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if not alpha_chars:
            return 0.0

        ratio = self._get_vowel_ratio(text)
        min_ratio = self.kwargs.get("min_vowel_ratio", 0.15)
        max_ratio = self.kwargs.get("max_vowel_ratio", 0.65)

        ratio_score = 0.0
        if ratio < min_ratio:
            ratio_score = (min_ratio - ratio) / min_ratio
        elif ratio > max_ratio:
            ratio_score = (ratio - max_ratio) / (1.0 - max_ratio)

        max_consonant_run = self._get_max_consonant_run(text)
        cluster_threshold = self.kwargs.get("consonant_cluster_len", 4)
        cluster_score = 0.0
        if max_consonant_run >= cluster_threshold:
            cluster_score = min((max_consonant_run - cluster_threshold + 1) / 4, 1.0)

        return min(max(ratio_score, cluster_score), 1.0)

