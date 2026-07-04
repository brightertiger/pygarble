# -*- coding: utf-8 -*-
"""
Mojibake detection strategy for garble detection.

Detects encoding corruption patterns (UTF-8 decoded as Latin-1, etc.).
"""

import unicodedata
from typing import Any, List, Tuple

from .base import BaseStrategy


# Common mojibake byte patterns: (corrupted bytes, description)
# These occur when UTF-8 is incorrectly decoded as Latin-1 or Windows-1252
MOJIBAKE_BYTE_PATTERNS: List[Tuple[bytes, str]] = [
    # UTF-8 decoded as Latin-1 (accented characters)
    (b"\xc3\xa1", "a-acute"),
    (b"\xc3\xa9", "e-acute"),
    (b"\xc3\xad", "i-acute"),
    (b"\xc3\xb3", "o-acute"),
    (b"\xc3\xba", "u-acute"),
    (b"\xc3\xb1", "n-tilde"),
    (b"\xc3\xbc", "u-umlaut"),
    (b"\xc3\xb6", "o-umlaut"),
    (b"\xc3\xa4", "a-umlaut"),
    (b"\xc3\xa7", "c-cedilla"),
    # Common pattern starters
    (b"\xc3\x82", "A-circumflex"),
    (b"\xc3\x83", "A-tilde"),
    (b"\xc2\xa0", "nbsp"),
    # Smart quote mojibake
    (b"\xe2\x80\x99", "right-single-quote"),
    (b"\xe2\x80\x9c", "left-double-quote"),
    (b"\xe2\x80\x9d", "right-double-quote"),
    (b"\xe2\x80\x94", "em-dash"),
    (b"\xe2\x80\x93", "en-dash"),
]


class MojibakeStrategy(BaseStrategy):
    """
    Detect garbled text caused by encoding corruption (mojibake).

    Mojibake occurs when text is decoded with the wrong encoding,
    producing garbled characters. This commonly happens when UTF-8
    text is decoded as Latin-1 or Windows-1252.

    This strategy detects common mojibake patterns without
    requiring external dependencies.

    Parameters
    ----------
    pattern_threshold : int, optional
        Number of mojibake patterns found to flag as garbled.
        Default is 1 (any mojibake pattern triggers detection).

    ratio_threshold : float, optional
        Ratio of suspicious characters to total length above
        which text is considered garbled. Default is 0.05.

    check_replacement_char : bool, optional
        Whether to check for Unicode replacement character.
        Default is True.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.MOJIBAKE)
    >>> # Text with mojibake (UTF-8 bytes interpreted as Latin-1)
    >>> detector.predict_proba("hello") < 0.5  # Clean text
    True
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.pattern_threshold = kwargs.get("pattern_threshold", 1)
        self.ratio_threshold = kwargs.get("ratio_threshold", 0.05)
        self.check_replacement_char = kwargs.get("check_replacement_char", True)

        if self.pattern_threshold < 1:
            raise ValueError("pattern_threshold must be at least 1")
        if not 0.0 <= self.ratio_threshold <= 1.0:
            raise ValueError("ratio_threshold must be between 0.0 and 1.0")

        # Convert byte patterns to string patterns for matching.
        # Real-world mojibake usually comes from misdecoding as cp1252
        # (e.g. \xe2\x80\x99 -> "\u00e2\u20ac\u2122"), but keep the latin-1 variants
        # (C1 controls) too since both appear in the wild.
        self._mojibake_patterns: List[str] = []
        for pattern_bytes, _ in MOJIBAKE_BYTE_PATTERNS:
            for encoding in ("cp1252", "latin-1"):
                try:
                    decoded = pattern_bytes.decode(encoding)
                except Exception:
                    continue
                if decoded not in self._mojibake_patterns:
                    self._mojibake_patterns.append(decoded)

        self._replacement_char = "\ufffd"  # Unicode replacement character

        # Lead characters of typical mojibake sequences (UTF-8 lead
        # bytes 0xC2/0xC3/0xD0/0xD1/0xE2 misdecoded as single chars)
        self._mojibake_lead_chars = frozenset("\xc2\xc3\xd0\xd1\xe2")

    def _count_mojibake_patterns(self, text: str) -> int:
        """Count occurrences of known mojibake patterns."""
        count = 0
        for pattern in self._mojibake_patterns:
            count += text.count(pattern)
        return count

    def _count_replacement_chars(self, text: str) -> int:
        """Count Unicode replacement characters."""
        return text.count(self._replacement_char)

    def _is_mojibake_follow_char(self, char: str) -> bool:
        """Check if a character can be the tail of a mojibake sequence."""
        code = ord(char)
        if 0x80 <= code <= 0xFF:
            return True
        # cp1252-misdecoded continuation bytes surface as punctuation
        # and symbol characters (e.g. "€", "™", "œ" in "â€™").
        if code > 0xFF:
            category = unicodedata.category(char)
            return category.startswith("P") or category.startswith("S")
        return False

    def _has_high_byte_density(self, text: str) -> float:
        """
        Check for high density of mojibake-shaped character sequences.

        Standalone accented letters ("café", "naïve") are legitimate.
        Only count high-byte characters that appear in mojibake-shaped
        sequences: a lead char (Ã/Â/â/Ð/Ñ) followed by another
        high-byte, C1-control, or punctuation/symbol character - plus
        bare C1 control characters, which never occur in clean text.
        """
        if not text:
            return 0.0

        suspicious_count = 0
        for i, char in enumerate(text):
            code = ord(char)
            # C1 controls are suspicious on their own
            if 0x80 <= code <= 0x9F:
                suspicious_count += 1
                continue
            if char in self._mojibake_lead_chars and i + 1 < len(text):
                if self._is_mojibake_follow_char(text[i + 1]):
                    # Count the lead and its follower
                    suspicious_count += 2

        suspicious_count = min(suspicious_count, len(text))
        return suspicious_count / len(text)

    def _check_double_encoding(self, text: str) -> bool:
        """
        Check for signs of double UTF-8 encoding.

        Double encoding produces patterns like "Ã�Â" which are
        very distinctive mojibake signatures.
        """
        # Common double-encoding signatures (latin-1 and cp1252 forms)
        double_encode_sigs = [
            "\xc3\x83\xc2",    # Double-encoded UTF-8 start (latin-1)
            "\xc3\x82\xc2",    # Another double-encoding pattern (latin-1)
            "\xc3ƒ\xc2",  # "ÃƒÂ" - cp1252 double-encoding
            "\xc3‚\xc2",  # "Ã‚Â" - cp1252 double-encoding
        ]
        for sig in double_encode_sigs:
            if sig in text:
                return True
        return False

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on mojibake detection.
        """
        if not text or len(text) < 3:
            return 0.0

        scores = []

        # Check for known mojibake patterns
        pattern_count = self._count_mojibake_patterns(text)
        if pattern_count >= self.pattern_threshold:
            # More patterns = higher confidence
            pattern_score = min(1.0, 0.7 + (pattern_count * 0.1))
            scores.append(pattern_score)

        # Check for replacement characters
        if self.check_replacement_char:
            replacement_count = self._count_replacement_chars(text)
            if replacement_count > 0:
                # Any replacement char is a strong signal
                replacement_score = min(1.0, 0.8 + (replacement_count * 0.05))
                scores.append(replacement_score)

        # Check high-byte density
        byte_density = self._has_high_byte_density(text)
        if byte_density >= self.ratio_threshold:
            # Map density to score
            density_score = min(1.0, byte_density * 5)
            scores.append(density_score)

        # Check for double encoding (very strong signal)
        if self._check_double_encoding(text):
            scores.append(0.95)

        # Return maximum score (any strong signal is sufficient)
        return max(scores) if scores else 0.0
