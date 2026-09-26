import re
from typing import Any

from ...data import ENGLISH_WORDS
from ...validation import parameter_value
from .base import BaseStrategy

_WORD_PATTERN = re.compile(r"[a-z0-9]+")

_LAYOUTS = {
    "qwerty": ("1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"),
    "azerty": ("1234567890", "azertyuiop", "qsdfghjklm", "wxcvbn"),
    "qwertz": ("1234567890", "qwertzuiop", "asdfghjkl", "yxcvbnm"),
}
_QWERTY_ROWS = _LAYOUTS["qwerty"]


def _build_adjacency(rows: tuple = _QWERTY_ROWS) -> dict:
    offsets = (0.0, 0.25, 0.5, 1.0)
    coordinates = {
        char: (column + offsets[row], row)
        for row, keys in enumerate(rows)
        for column, char in enumerate(keys)
    }
    return {
        char: {
            other
            for other, (ox, oy) in coordinates.items()
            if other != char and abs(x - ox) <= 1.1 and abs(y - oy) <= 1
        }
        for char, (x, y) in coordinates.items()
    }


class KeyboardAdjacencyStrategy(BaseStrategy):
    """Detect keyboard mashing via physical key-adjacency walks.

    Keyboard mash ("asdfgh", "qweasd") consists of runs of physically
    adjacent keys far longer than English produces. This measures, per
    word, the longest chain of consecutive adjacent-or-repeated keys and
    the longest single-row run. Dictionary words are exempt (English has
    pathological cases like "typewriter" - entirely top-row).

    Args:
        min_word_length: shortest word to analyze (default 5)
        chain_threshold: adjacent-key chain length that flags a word
            (default 6; real English words peak at 5 - "udder", "ponder")
        row_run_threshold: same-row run length that flags a word when the
            word is also vowel-poor (default 6; the run alone is not
            enough - the vowel-rich top row gives real words runs up to 8,
            e.g. "xmlhttprequest")

    Example:
        >>> detector = GarbleDetector(Strategy.KEYBOARD_ADJACENCY)
        >>> detector.predict("asdfgh jklqwe")
        True
        >>> detector.predict("typewriter repairs")
        False
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        layout = kwargs.get("keyboard_layout", "qwerty")
        if layout not in _LAYOUTS:
            raise ValueError(
                "keyboard_layout must be qwerty, azerty, or qwertz"
            )
        self.rows = _LAYOUTS[layout]
        self.adjacent = _build_adjacency(self.rows)
        self.row_of = {
            char: idx for idx, row in enumerate(self.rows) for char in row
        }
        self.min_word_length: int = parameter_value(
            "min_word_length", kwargs.get("min_word_length", 5), 5
        )
        self.chain_threshold: int = parameter_value(
            "chain_threshold", kwargs.get("chain_threshold", 6), 6
        )
        self.row_run_threshold: int = parameter_value(
            "row_run_threshold", kwargs.get("row_run_threshold", 6), 6
        )

    def _longest_adjacency_chain(self, word: str) -> int:
        longest = current = 1
        for prev, char in zip(word, word[1:]):
            if char == prev or char in self.adjacent.get(prev, ()):
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def _longest_row_run(self, word: str) -> int:
        longest = current = 1
        for prev, char in zip(word, word[1:]):
            if self.row_of.get(char) is not None and self.row_of.get(
                char
            ) == self.row_of.get(prev):
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def _is_mashed(self, word: str) -> bool:
        if len(word) < self.min_word_length:
            return False
        if sum(char.isalpha() for char in word) < 3:
            return False
        # Whole straight row walks remain suspicious even if a web-frequency
        # list contains them; ordinary row words such as typewriter are safe.
        if len(word) >= 5 and any(
            word in row or word in row[::-1] for row in self.rows
        ):
            return True
        if word in ENGLISH_WORDS:
            return False
        chain = self._longest_adjacency_chain(word)
        if chain >= self.chain_threshold and chain / len(word) >= 0.6:
            return True
        vowels = sum(1 for c in word if c in "aeiou")
        return (
            vowels <= 1
            and self._longest_row_run(word) >= self.row_run_threshold
        )

    def _predict_proba_impl(self, text: str) -> float:
        folded = self._fold_diacritics(text).lower()
        words = _WORD_PATTERN.findall(folded)
        if not words:
            return 0.0
        eligible = [w for w in words if len(w) >= self.min_word_length]
        if not eligible:
            return 0.0
        mashed = sum(1 for w in eligible if self._is_mashed(w))
        if mashed == 0:
            return 0.0
        # Any mashed word is strong evidence; more of them raises confidence
        return min(1.0, 0.6 + 0.4 * mashed / len(eligible))
