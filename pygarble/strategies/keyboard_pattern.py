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
        alpha_text = "".join(c.lower() for c in text if c.isalpha())
        return [alpha_text[i:i+3] for i in range(len(alpha_text) - 2)]

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

    def _predict_impl(self, text: str) -> bool:
        keyboard_ratio = self._get_keyboard_pattern_ratio(text)
        keyboard_threshold = self.kwargs.get("keyboard_threshold", 0.3)

        if keyboard_ratio >= keyboard_threshold:
            return True

        common_ratio = self._get_common_trigram_ratio(text)
        common_threshold = self.kwargs.get("common_trigram_threshold", 0.1)

        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) >= 5 and common_ratio < common_threshold:
            return True

        if self._has_repeated_bigram_pattern(text):
            return True

        return False

    def _predict_proba_impl(self, text: str) -> float:
        keyboard_ratio = self._get_keyboard_pattern_ratio(text)
        common_ratio = self._get_common_trigram_ratio(text)

        keyboard_score = min(keyboard_ratio / 0.3, 1.0)

        common_score = 0.0
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) >= 5:
            common_score = max(0, 1.0 - (common_ratio / 0.15))

        repeated_score = 0.5 if self._has_repeated_bigram_pattern(text) else 0.0

        return min(max(keyboard_score, common_score * 0.7, repeated_score), 1.0)

