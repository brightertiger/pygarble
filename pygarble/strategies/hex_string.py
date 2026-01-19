"""
Hexadecimal string detection strategy for garble detection.

Detects hash strings, UUIDs, and other hexadecimal content.
"""

import re
from typing import Any

from .base import BaseStrategy


class HexStringStrategy(BaseStrategy):
    """
    Detect garbled text that appears to be hexadecimal data.

    This strategy identifies:
    - MD5 hashes (32 hex chars)
    - SHA256 hashes (64 hex chars)
    - UUIDs (8-4-4-4-12 format)
    - Generic hex strings
    - Base64-like patterns

    Parameters
    ----------
    min_hex_length : int, optional
        Minimum length of hex sequence to detect. Default is 16.

    hex_ratio_threshold : float, optional
        Ratio of hex characters above which text is suspicious.
        Default is 0.7.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.HEX_STRING)
    >>> detector.predict("5d41402abc4b2a76b9719d911017c592")
    True
    >>> detector.predict("hello world")
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_hex_length = kwargs.get("min_hex_length", 16)
        self.hex_ratio_threshold = kwargs.get("hex_ratio_threshold", 0.7)

        if self.min_hex_length < 0:
            raise ValueError("min_hex_length must be non-negative")
        if not 0.0 <= self.hex_ratio_threshold <= 1.0:
            raise ValueError("hex_ratio_threshold must be between 0.0 and 1.0")

        # Patterns for various hex formats
        self._md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
        self._sha256_pattern = re.compile(r"^[a-fA-F0-9]{64}$")
        self._uuid_pattern = re.compile(
            r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
        )
        self._long_hex_pattern = re.compile(r"[a-fA-F0-9]{16,}")
        self._base64_pattern = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")

    def _is_pure_hash(self, text: str) -> bool:
        """Check if text is a pure hash string."""
        text = text.strip()
        return bool(
            self._md5_pattern.match(text)
            or self._sha256_pattern.match(text)
        )

    def _contains_uuid(self, text: str) -> bool:
        """Check if text contains a UUID."""
        return bool(self._uuid_pattern.search(text))

    def _contains_long_hex(self, text: str) -> float:
        """
        Check for long hexadecimal sequences.

        Returns score based on length of hex sequences found.
        """
        matches = self._long_hex_pattern.findall(text)
        if not matches:
            return 0.0

        # Score based on longest match
        max_len = max(len(m) for m in matches)
        if max_len >= 64:
            return 1.0
        elif max_len >= 32:
            return 0.9
        elif max_len >= 16:
            return 0.7
        return 0.5

    def _is_base64_like(self, text: str) -> bool:
        """Check if text looks like base64."""
        text = text.strip()
        return bool(self._base64_pattern.match(text))

    def _compute_hex_ratio(self, text: str) -> float:
        """Compute ratio of hex characters in text."""
        hex_chars = set("0123456789abcdefABCDEF")
        alnum_chars = [c for c in text if c.isalnum()]

        if len(alnum_chars) < self.min_hex_length:
            return 0.0

        hex_count = sum(1 for c in alnum_chars if c in hex_chars)
        return hex_count / len(alnum_chars)

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on hex content analysis.
        """
        if not text or len(text) < 8:
            return 0.0

        # Check for pure hash strings
        if self._is_pure_hash(text):
            return 1.0

        # Check for base64-like content
        if self._is_base64_like(text):
            return 0.9

        # Check for UUID (but in context, might be OK)
        if self._contains_uuid(text):
            # If text is mostly UUID, it's garbled
            # If UUID is embedded in normal text, might be OK
            words = text.split()
            if len(words) <= 2:
                return 0.8

        # Check for long hex sequences
        hex_score = self._contains_long_hex(text)
        if hex_score > 0:
            return hex_score

        # Check overall hex ratio
        hex_ratio = self._compute_hex_ratio(text)
        if hex_ratio >= self.hex_ratio_threshold:
            denominator = 1.0 - self.hex_ratio_threshold
            if denominator > 0:
                return 0.6 + 0.4 * ((hex_ratio - self.hex_ratio_threshold) / denominator)
            else:
                return 1.0  # threshold is 1.0, so any match is max score

        return 0.0
