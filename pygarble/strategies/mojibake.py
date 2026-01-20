# -*- coding: utf-8 -*-
"""
Mojibake detection strategy for garble detection.

Detects encoding corruption patterns (UTF-8 decoded as Latin-1, etc.).
"""

import re
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

        # Convert byte patterns to string patterns for matching
        self._mojibake_patterns: List[str] = []
        for pattern_bytes, _ in MOJIBAKE_BYTE_PATTERNS:
            try:
                # Decode as Latin-1 to get the mojibake string representation
                self._mojibake_patterns.append(pattern_bytes.decode('latin-1'))
            except Exception:
                pass

        self._replacement_char = "\ufffd"  # Unicode replacement character

        # Regex for detecting high concentration of Latin-1 supplement chars
        self._high_byte_pattern = re.compile(r'[\x80-\xff]')

    def _count_mojibake_patterns(self, text: str) -> int:
        """Count occurrences of known mojibake patterns."""
        count = 0
        for pattern in self._mojibake_patterns:
            count += text.count(pattern)
        return count

    def _count_replacement_chars(self, text: str) -> int:
        """Count Unicode replacement characters."""
        return text.count(self._replacement_char)

    def _has_high_byte_density(self, text: str) -> float:
        """
        Check for high density of characters in the Latin-1 supplement range.

        These characters (U+0080 to U+00FF) often indicate mojibake
        when they appear frequently in supposedly English text.
        """
        if not text:
            return 0.0

        # Count chars in Latin-1 supplement range
        matches = self._high_byte_pattern.findall(text)
        suspicious_count = len(matches)

        # Skip if the suspicious chars are legitimate
        # (copyright, registered trademark, degree, etc.)
        legitimate_chars = {'\xa9', '\xae', '\xb0', '\xb1', '\xb7'}
        for char in legitimate_chars:
            suspicious_count -= text.count(char)

        suspicious_count = max(0, suspicious_count)
        return suspicious_count / len(text) if text else 0.0

    def _check_double_encoding(self, text: str) -> bool:
        """
        Check for signs of double UTF-8 encoding.

        Double encoding produces patterns like "Ã�Â" which are
        very distinctive mojibake signatures.
        """
        # Common double-encoding signatures
        double_encode_sigs = [
            "\xc3\x83\xc2",  # Double-encoded UTF-8 start
            "\xc3\x82\xc2",  # Another double-encoding pattern
        ]
        for sig in double_encode_sigs:
            if sig in text:
                return True
        return False

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

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
