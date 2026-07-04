from typing import FrozenSet

from .base import BaseStrategy


VOWELS = frozenset("aeiou")
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")

# All-uppercase words up to this length are treated as acronyms and
# skipped; longer uppercase words (e.g. shouted gibberish) are analyzed
# in lowercase instead of being silently dropped.
MAX_ACRONYM_LENGTH = 6


class VowelRatioStrategy(BaseStrategy):
    def _filter_acronyms(self, text: str) -> str:
        """Skip short all-uppercase words (likely acronyms).

        Longer all-uppercase words are lowercased and analyzed normally so
        uppercase gibberish ("QWRTPZXCV") is not exempted wholesale.
        """
        words = text.split()
        filtered = []
        for w in words:
            if len(w) > 1 and w.isalpha() and w.isupper():
                if len(w) <= MAX_ACRONYM_LENGTH:
                    continue
                filtered.append(w.lower())
            else:
                filtered.append(w)
        return " ".join(filtered)

    def _word_vowels(self, word: str) -> FrozenSet[str]:
        """Vowel set for a word: 'y' acts as a vowel in words that would
        otherwise be vowelless (my, gym, rhythm, sky)."""
        if any(c in VOWELS for c in word):
            return VOWELS
        return VOWELS | frozenset("y")

    def _get_vowel_ratio(self, text: str) -> float:
        vowel_count = 0
        total = 0
        for word in text.lower().split():
            vowels = self._word_vowels(word)
            for c in word:
                if c.isalpha():
                    total += 1
                    if c in vowels:
                        vowel_count += 1

        if total == 0:
            return 0.0
        return vowel_count / total

    def _has_consonant_cluster(self, text: str) -> bool:
        cluster_len = self.kwargs.get("consonant_cluster_len", 4)
        return self._get_max_consonant_run(text) >= cluster_len

    def _get_max_consonant_run(self, text: str) -> int:
        max_run = 0
        for word in text.lower().split():
            vowels = self._word_vowels(word)
            current_run = 0
            for c in word:
                if c.isalpha() and c not in vowels:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
        return max_run

    def _predict_proba_impl(self, text: str) -> float:
        text = self._filter_acronyms(text)
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if not alpha_chars:
            return 0.0

        ratio = self._get_vowel_ratio(text)
        min_ratio = self.kwargs.get("min_vowel_ratio", 0.15)
        max_ratio = self.kwargs.get("max_vowel_ratio", 0.65)

        ratio_score = 0.0
        if ratio < min_ratio:
            if min_ratio > 0:
                ratio_score = (min_ratio - ratio) / min_ratio
            else:
                ratio_score = 0.0  # Can't be below 0
        elif ratio > max_ratio:
            denominator = 1.0 - max_ratio
            if denominator > 0:
                ratio_score = (ratio - max_ratio) / denominator
            else:
                ratio_score = 1.0  # max_ratio is 1.0, ratio > 1.0 is impossible

        max_consonant_run = self._get_max_consonant_run(text)
        cluster_threshold = self.kwargs.get("consonant_cluster_len", 4)
        cluster_score = 0.0
        if max_consonant_run >= cluster_threshold:
            cluster_score = min((max_consonant_run - cluster_threshold) / 4, 1.0)

        return min(max(ratio_score, cluster_score), 1.0)
