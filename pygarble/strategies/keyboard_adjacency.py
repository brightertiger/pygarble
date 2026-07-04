import re

from ..data import ENGLISH_WORDS
from .base import BaseStrategy

_WORD_PATTERN = re.compile(r"[a-z]+")

_QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _build_adjacency():
    """Physical neighbor map for QWERTY, including diagonal neighbors
    between staggered rows."""
    adjacency = {}
    for row_idx, row in enumerate(_QWERTY_ROWS):
        for col, char in enumerate(row):
            neighbors = set()
            if col > 0:
                neighbors.add(row[col - 1])
            if col < len(row) - 1:
                neighbors.add(row[col + 1])
            # Staggered rows: key at col sits between cols (col) and
            # (col + 1) of the row above, and (col - 1)/(col) below.
            if row_idx > 0:
                above = _QWERTY_ROWS[row_idx - 1]
                for offset in (0, 1):
                    if 0 <= col + offset < len(above):
                        neighbors.add(above[col + offset])
            if row_idx < len(_QWERTY_ROWS) - 1:
                below = _QWERTY_ROWS[row_idx + 1]
                for offset in (-1, 0):
                    if 0 <= col + offset < len(below):
                        neighbors.add(below[col + offset])
            adjacency[char] = neighbors
    return adjacency


_ADJACENT = _build_adjacency()
_ROW_OF = {
    char: idx for idx, row in enumerate(_QWERTY_ROWS) for char in row
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_word_length = kwargs.get("min_word_length", 5)
        self.chain_threshold = kwargs.get("chain_threshold", 6)
        self.row_run_threshold = kwargs.get("row_run_threshold", 6)

    def _longest_adjacency_chain(self, word: str) -> int:
        longest = current = 1
        for prev, char in zip(word, word[1:]):
            if char == prev or char in _ADJACENT.get(prev, ()):
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def _longest_row_run(self, word: str) -> int:
        longest = current = 1
        for prev, char in zip(word, word[1:]):
            if _ROW_OF.get(char) is not None and _ROW_OF.get(
                char
            ) == _ROW_OF.get(prev):
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def _is_mashed(self, word: str) -> bool:
        if len(word) < self.min_word_length:
            return False
        if word in ENGLISH_WORDS:
            return False
        if self._longest_adjacency_chain(word) >= self.chain_threshold:
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
