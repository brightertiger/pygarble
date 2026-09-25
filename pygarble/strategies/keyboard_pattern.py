import re
from typing import List, Set

from .base import BaseStrategy

KEYBOARD_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]

KEYBOARD_SEQUENCES: Set[str] = set()
for row in KEYBOARD_ROWS:
    for i in range(len(row) - 2):
        KEYBOARD_SEQUENCES.add(row[i : i + 3])
        KEYBOARD_SEQUENCES.add(row[i : i + 3][::-1])

COMMON_TRIGRAMS: Set[str] = {
    "the",
    "and",
    "ing",
    "ion",
    "tio",
    "ent",
    "ati",
    "for",
    "her",
    "ter",
    "hat",
    "tha",
    "ere",
    "ate",
    "his",
    "con",
    "res",
    "ver",
    "all",
    "ons",
    "nce",
    "men",
    "ith",
    "ted",
    "ers",
    "pro",
    "thi",
    "wit",
    "are",
    "ess",
    "not",
    "ive",
    "was",
    "ect",
    "rea",
    "com",
    "eve",
    "per",
    "int",
    "est",
    "sta",
    "cti",
    "ica",
    "ist",
    "ear",
    "ain",
    "one",
    "our",
    "iti",
    "rat",
}

_REPEATED_BIGRAM = re.compile(r"(..)\1{2,}")


class KeyboardPatternStrategy(BaseStrategy):
    def _get_trigrams(self, text: str) -> List[str]:
        return self._trigrams(self._novel_words(text))

    @staticmethod
    def _trigrams(words: List[str]) -> List[str]:
        # Per word so trigrams never span word boundaries.
        trigrams: List[str] = []
        for word in words:
            trigrams.extend(word[i : i + 3] for i in range(len(word) - 2))
        return trigrams

    def _get_keyboard_pattern_ratio(self, text: str) -> float:
        return self._keyboard_ratio(self._get_trigrams(text))

    @staticmethod
    def _keyboard_ratio(trigrams: List[str]) -> float:
        if not trigrams:
            return 0.0
        return sum(1 for tg in trigrams if tg in KEYBOARD_SEQUENCES) / len(
            trigrams
        )

    def _get_common_trigram_ratio(self, text: str) -> float:
        return self._common_ratio(self._get_trigrams(text))

    @staticmethod
    def _common_ratio(trigrams: List[str]) -> float:
        if not trigrams:
            return 0.0
        return sum(1 for tg in trigrams if tg in COMMON_TRIGRAMS) / len(
            trigrams
        )

    def _has_repeated_bigram_pattern(self, text: str) -> bool:
        return self._repeated_bigram(self._novel_words(text))

    @staticmethod
    def _repeated_bigram(words: List[str]) -> bool:
        # Judged inside a single novel word; "go go go" is English.
        return any(
            len(word) >= 6 and _REPEATED_BIGRAM.search(word) for word in words
        )

    def _predict_proba_impl(self, text: str) -> float:
        words = self._novel_words(text)
        trigrams = self._trigrams(words)
        keyboard_score = min(self._keyboard_ratio(trigrams) / 0.3, 1.0)

        # The common-trigram deficit only means something when there are
        # enough trigrams to expect hits from a 50-item list.
        common_score = 0.0
        if len(trigrams) >= 15:
            confidence = min(1.0, len(trigrams) / 28.0)
            common_score = (
                max(0.0, 1.0 - (self._common_ratio(trigrams) / 0.15))
                * confidence
            )

        repeated_score = 0.5 if self._repeated_bigram(words) else 0.0
        return min(
            max(keyboard_score, common_score * 0.7, repeated_score), 1.0
        )
