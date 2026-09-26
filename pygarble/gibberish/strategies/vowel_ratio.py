from typing import Any, FrozenSet

from ...validation import positive_int, unit_interval
from .base import BaseStrategy

VOWELS = frozenset("aeiou")
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")

# All-uppercase words up to this length are treated as acronyms and
# skipped; longer uppercase words (e.g. shouted gibberish) are analyzed
# in lowercase instead of being silently dropped.
MAX_ACRONYM_LENGTH = 6


class VowelRatioStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_vowel_ratio = unit_interval(
            "min_vowel_ratio", kwargs.get("min_vowel_ratio", 0.15)
        )
        self.max_vowel_ratio = unit_interval(
            "max_vowel_ratio", kwargs.get("max_vowel_ratio", 0.65)
        )
        if self.min_vowel_ratio > self.max_vowel_ratio:
            raise ValueError("min_vowel_ratio must not exceed max_vowel_ratio")
        self.consonant_cluster_len = positive_int(
            "consonant_cluster_len", kwargs.get("consonant_cluster_len", 4)
        )
        # Ratios on one to three letters are noise ("I", "Hmm", "Mr. Ng").
        self.min_length = positive_int(
            "min_length", kwargs.get("min_length", 4)
        )

    def _letters(self, text: str) -> int:
        """Letters in words of three or more letters; "Mr", "Ng",
        "I" are abbreviations, not evidence."""
        total = 0
        for word in self._filter_acronyms(text).split():
            count = sum(1 for c in word if c.isalpha())
            if count >= 3:
                total += count
        return total

    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return self._letters(text) >= self.min_length

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
            if sum(1 for c in word if c.isalpha()) < 3:
                continue
            vowels = self._word_vowels(word)
            for c in word:
                if c.isalpha():
                    total += 1
                    if c in vowels:
                        vowel_count += 1

        if total == 0:
            return 0.0
        return vowel_count / total

    def _get_max_consonant_run(self, text: str) -> int:
        max_run = 0
        for word in text.lower().split():
            if sum(1 for c in word if c.isalpha()) < 3:
                continue
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
        if self._letters(text) < self.min_length:
            return 0.0
        ratio = self._get_vowel_ratio(text)
        if ratio == 0.0 and self._get_max_consonant_run(text) == 0:
            return 0.0  # every word was a short abbreviation

        ratio_score = 0.0
        if ratio < self.min_vowel_ratio:
            ratio_score = (self.min_vowel_ratio - ratio) / self.min_vowel_ratio
        elif ratio > self.max_vowel_ratio:
            denominator = 1.0 - self.max_vowel_ratio
            ratio_score = (
                (ratio - self.max_vowel_ratio) / denominator
                if denominator > 0
                else 1.0
            )

        run = self._get_max_consonant_run(text)
        cluster_score = 0.0
        if run >= self.consonant_cluster_len:
            cluster_score = min((run - self.consonant_cluster_len) / 4, 1.0)

        return min(max(ratio_score, cluster_score), 1.0)
