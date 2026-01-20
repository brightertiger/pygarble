"""
Unicode script mixing detection strategy for garble detection.

Detects suspicious mixing of different Unicode scripts (e.g., Cyrillic
characters mixed with Latin) which is common in spam and phishing.
"""

import unicodedata
from collections import Counter
from typing import Any, Dict, Set

from .base import BaseStrategy


# Common homoglyphs: characters that look like Latin but are from other scripts
# Format: {lookalike_char: (latin_equivalent, script_name)}
HOMOGLYPHS: Dict[str, tuple] = {
    # Cyrillic homoglyphs
    "а": ("a", "Cyrillic"),  # Cyrillic small a
    "А": ("A", "Cyrillic"),  # Cyrillic capital A
    "В": ("B", "Cyrillic"),  # Cyrillic capital Ve
    "с": ("c", "Cyrillic"),  # Cyrillic small Es
    "С": ("C", "Cyrillic"),  # Cyrillic capital Es
    "е": ("e", "Cyrillic"),  # Cyrillic small Ie
    "Е": ("E", "Cyrillic"),  # Cyrillic capital Ie
    "ѐ": ("e", "Cyrillic"),  # Cyrillic small Ie with grave
    "Н": ("H", "Cyrillic"),  # Cyrillic capital En
    "і": ("i", "Cyrillic"),  # Cyrillic small Byelorussian-Ukrainian I
    "І": ("I", "Cyrillic"),  # Cyrillic capital Byelorussian-Ukrainian I
    "ј": ("j", "Cyrillic"),  # Cyrillic small Je
    "К": ("K", "Cyrillic"),  # Cyrillic capital Ka
    "М": ("M", "Cyrillic"),  # Cyrillic capital Em
    "о": ("o", "Cyrillic"),  # Cyrillic small O
    "О": ("O", "Cyrillic"),  # Cyrillic capital O
    "р": ("p", "Cyrillic"),  # Cyrillic small Er
    "Р": ("P", "Cyrillic"),  # Cyrillic capital Er
    "ѕ": ("s", "Cyrillic"),  # Cyrillic small Dze
    "Ѕ": ("S", "Cyrillic"),  # Cyrillic capital Dze
    "Т": ("T", "Cyrillic"),  # Cyrillic capital Te
    "у": ("y", "Cyrillic"),  # Cyrillic small U
    "х": ("x", "Cyrillic"),  # Cyrillic small Ha
    "Х": ("X", "Cyrillic"),  # Cyrillic capital Ha
    # Greek homoglyphs
    "Α": ("A", "Greek"),  # Greek capital Alpha
    "Β": ("B", "Greek"),  # Greek capital Beta
    "Ε": ("E", "Greek"),  # Greek capital Epsilon
    "Ζ": ("Z", "Greek"),  # Greek capital Zeta
    "Η": ("H", "Greek"),  # Greek capital Eta
    "Ι": ("I", "Greek"),  # Greek capital Iota
    "Κ": ("K", "Greek"),  # Greek capital Kappa
    "Μ": ("M", "Greek"),  # Greek capital Mu
    "Ν": ("N", "Greek"),  # Greek capital Nu
    "Ο": ("O", "Greek"),  # Greek capital Omicron
    "ο": ("o", "Greek"),  # Greek small Omicron
    "Ρ": ("P", "Greek"),  # Greek capital Rho
    "Τ": ("T", "Greek"),  # Greek capital Tau
    "Υ": ("Y", "Greek"),  # Greek capital Upsilon
    "Χ": ("X", "Greek"),  # Greek capital Chi
    "ν": ("v", "Greek"),  # Greek small Nu
    "ι": ("i", "Greek"),  # Greek small Iota
    # Other confusables
    "ⅰ": ("i", "Number Forms"),  # Roman numeral one
    "ⅱ": ("ii", "Number Forms"),  # Roman numeral two
    "ⅲ": ("iii", "Number Forms"),  # Roman numeral three
    "ℓ": ("l", "Letterlike"),  # Script small L
    "№": ("No", "Letterlike"),  # Numero sign
}

# Scripts that are commonly mixed legitimately (don't flag)
COMPATIBLE_SCRIPTS: Set[frozenset] = {
    frozenset({"Latin", "Common"}),
    frozenset({"Latin", "Common", "Inherited"}),
}


class UnicodeScriptStrategy(BaseStrategy):
    """
    Detect garbled/suspicious text based on Unicode script mixing.

    This strategy identifies:
    - Cyrillic characters disguised as Latin (homoglyphs)
    - Greek characters mixed into Latin text
    - Suspicious script combinations (potential spoofing)

    Parameters
    ----------
    homoglyph_threshold : int, optional
        Number of homoglyphs to flag as suspicious.
        Default is 1.

    max_scripts : int, optional
        Maximum number of different scripts allowed.
        Default is 2 (Latin + one other is usually OK).

    check_homoglyphs : bool, optional
        Whether to check for known homoglyphs.
        Default is True.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
    >>> detector.predict("pаypal")  # 'а' is Cyrillic
    True
    >>> detector.predict("paypal")  # All Latin
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.homoglyph_threshold = kwargs.get("homoglyph_threshold", 1)
        self.max_scripts = kwargs.get("max_scripts", 2)
        self.check_homoglyphs = kwargs.get("check_homoglyphs", True)

        if self.homoglyph_threshold < 1:
            raise ValueError("homoglyph_threshold must be at least 1")
        if self.max_scripts < 1:
            raise ValueError("max_scripts must be at least 1")

    def _get_script(self, char: str) -> str:
        """Get the Unicode script for a character."""
        try:
            # Get the character name and extract script
            name = unicodedata.name(char, "")

            # Check common script prefixes
            if name.startswith("LATIN"):
                return "Latin"
            elif name.startswith("CYRILLIC"):
                return "Cyrillic"
            elif name.startswith("GREEK"):
                return "Greek"
            elif name.startswith("ARABIC"):
                return "Arabic"
            elif name.startswith("HEBREW"):
                return "Hebrew"
            elif name.startswith("CJK") or name.startswith("HANGUL"):
                return "CJK"
            elif name.startswith("HIRAGANA") or name.startswith("KATAKANA"):
                return "Japanese"
            elif char.isascii():
                if char.isalpha():
                    return "Latin"
                else:
                    return "Common"
            else:
                # For other characters, use category
                category = unicodedata.category(char)
                if category.startswith("L"):  # Letter
                    return "Unknown"
                elif category.startswith("N"):  # Number
                    return "Common"
                elif category.startswith("P") or category.startswith("S"):
                    return "Common"  # Punctuation/Symbol
                else:
                    return "Common"
        except ValueError:
            return "Unknown"

    def _count_homoglyphs(self, text: str) -> Dict[str, int]:
        """Count homoglyph characters by script."""
        counts: Dict[str, int] = Counter()
        for char in text:
            if char in HOMOGLYPHS:
                _, script = HOMOGLYPHS[char]
                counts[script] += 1
        return dict(counts)

    def _get_scripts_used(self, text: str) -> Set[str]:
        """Get set of all scripts used in text."""
        scripts = set()
        for char in text:
            if char.isalpha():  # Only check letters
                scripts.add(self._get_script(char))
        return scripts

    def _is_mixed_script_word(self, word: str) -> bool:
        """Check if a single word mixes scripts (very suspicious)."""
        alpha_chars = [c for c in word if c.isalpha()]
        if len(alpha_chars) < 2:
            return False

        scripts = set()
        for char in alpha_chars:
            script = self._get_script(char)
            if script not in {"Common", "Inherited", "Unknown"}:
                scripts.add(script)

        # A word mixing scripts is suspicious
        return len(scripts) > 1

    def _count_mixed_script_words(self, text: str) -> int:
        """Count words that mix different scripts."""
        words = text.split()
        return sum(1 for word in words if self._is_mixed_script_word(word))

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on script analysis.
        """
        if not text or len(text) < 2:
            return 0.0

        # Only analyze if text has alphabetic content
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) < 2:
            return 0.0

        scores = []

        # Check for homoglyphs
        if self.check_homoglyphs:
            homoglyph_counts = self._count_homoglyphs(text)
            total_homoglyphs = sum(homoglyph_counts.values())

            if total_homoglyphs >= self.homoglyph_threshold:
                # Homoglyphs are a strong signal of spoofing
                homoglyph_score = min(1.0, 0.8 + (total_homoglyphs * 0.05))
                scores.append(homoglyph_score)

        # Check for mixed-script words (very suspicious)
        mixed_word_count = self._count_mixed_script_words(text)
        if mixed_word_count > 0:
            # Mixed-script words are highly suspicious
            mixed_score = min(1.0, 0.7 + (mixed_word_count * 0.15))
            scores.append(mixed_score)

        # Check total scripts used
        scripts = self._get_scripts_used(text)
        # Remove Common and Inherited (they mix with everything)
        meaningful_scripts = scripts - {"Common", "Inherited", "Unknown"}

        if len(meaningful_scripts) > self.max_scripts:
            # Too many scripts is suspicious
            script_score = min(1.0, 0.5 + ((len(meaningful_scripts) - self.max_scripts) * 0.2))
            scores.append(script_score)

        # Return maximum score
        return max(scores) if scores else 0.0
