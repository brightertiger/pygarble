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
        KEYBOARD_SEQUENCES.add(row[i:i+3])
        KEYBOARD_SEQUENCES.add(row[i:i+3][::-1])

COMMON_TRIGRAMS: Set[str] = {
    "the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter",
    "hat", "tha", "ere", "ate", "his", "con", "res", "ver", "all", "ons",
    "nce", "men", "ith", "ted", "ers", "pro", "thi", "wit", "are", "ess",
    "not", "ive", "was", "ect", "rea", "com", "eve", "per", "int", "est",
    "sta", "cti", "ica", "ist", "ear", "ain", "one", "our", "iti", "rat",
}


class KeyboardPatternStrategy(BaseStrategy):
    def _get_trigrams(self, text: str) -> List[str]:
        # Judge only words the dictionary can't vouch for, per word so
        # trigrams never span word boundaries. Alliterative or rare-word
        # sentences ("she sells seashells") otherwise miss the common
        # trigram list and read as mash.
        trigrams: List[str] = []
        for word in self._novel_words(text):
            trigrams.extend(
                word[i:i + 3] for i in range(len(word) - 2)
            )
        return trigrams

    def _get_keyboard_pattern_ratio(self, text: str) -> float:
        trigrams = self._get_trigrams(text)
        if not trigrams:
            return 0.0

        keyboard_count = sum(1 for tg in trigrams if tg in KEYBOARD_SEQUENCES)
        return keyboard_count / len(trigrams)

    def _get_common_trigram_ratio(self, text: str) -> float:
        trigrams = self._get_trigrams(text)
        if not trigrams:
            return 0.0

        common_count = sum(1 for tg in trigrams if tg in COMMON_TRIGRAMS)
        return common_count / len(trigrams)

    def _has_repeated_bigram_pattern(self, text: str) -> bool:
        alpha_text = "".join(c.lower() for c in text if c.isalpha())
        if len(alpha_text) < 6:
            return False

        pattern = r"(..)(\1){2,}"
        return bool(re.search(pattern, alpha_text))

    def _predict_proba_impl(self, text: str) -> float:
        keyboard_ratio = self._get_keyboard_pattern_ratio(text)
        common_ratio = self._get_common_trigram_ratio(text)

        keyboard_score = min(keyboard_ratio / 0.3, 1.0)

        # The common-trigram deficit only means something when there are
        # enough trigrams to expect hits from a 50-item list; short texts
        # ("hello", "family dinner") must rely solely on the actual
        # keyboard-row-sequence checks. Scale the deficit in smoothly as
        # the amount of evidence grows.
        common_score = 0.0
        trigram_count = len(self._get_trigrams(text))
        if trigram_count >= 15:
            confidence = min(1.0, trigram_count / 28.0)
            common_score = max(0.0, 1.0 - (common_ratio / 0.15)) * confidence

        repeated_score = 0.5 if self._has_repeated_bigram_pattern(text) else 0.0

        return min(max(keyboard_score, common_score * 0.7, repeated_score), 1.0)

