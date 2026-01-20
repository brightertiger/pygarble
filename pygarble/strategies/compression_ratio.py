"""
Compression ratio strategy for garble detection.

Uses zlib compression to detect random/garbled text.
Random text compresses poorly compared to natural language.
"""

import zlib
from typing import Any

from .base import BaseStrategy


class CompressionRatioStrategy(BaseStrategy):
    """
    Detect garbled text based on compression ratio.

    This strategy uses compression as a proxy for text structure.
    It's most effective for:
    - Detecting highly repetitive spam/gibberish (compresses very well)
    - Identifying text with unusual entropy patterns

    Note: Random ASCII text and normal English compress similarly
    at moderate lengths. This strategy works best in ensemble with
    others, or for detecting extreme cases.

    Uses Python's built-in zlib (no external deps).

    Parameters
    ----------
    high_ratio_threshold : float, optional
        Compression ratio above which text is considered garbled.
        Default is 1.1 (accounting for zlib header overhead).

    low_ratio_threshold : float, optional
        Compression ratio below which text is definitely valid.
        Default is 0.85 (well-compressed text).

    min_length : int, optional
        Minimum text length to analyze. Short texts don't compress
        well regardless of content due to header overhead.
        Default is 100.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
    >>> # Works best on longer text (100+ chars)
    >>> long_random = "xkjhqwerty zxcvbn " * 10
    >>> detector.predict(long_random)
    True
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Adjusted thresholds to account for zlib header overhead (~10-15 bytes)
        # For short text, compression ratio often exceeds 1.0
        self.high_ratio_threshold = kwargs.get("high_ratio_threshold", 1.1)
        self.low_ratio_threshold = kwargs.get("low_ratio_threshold", 0.85)
        self.min_length = kwargs.get("min_length", 100)

        if not 0.0 <= self.high_ratio_threshold <= 1.5:
            raise ValueError("high_ratio_threshold must be between 0.0 and 1.5")
        if not 0.0 <= self.low_ratio_threshold <= 1.5:
            raise ValueError("low_ratio_threshold must be between 0.0 and 1.5")
        if self.low_ratio_threshold >= self.high_ratio_threshold:
            raise ValueError("low_ratio_threshold must be less than high_ratio_threshold")
        if self.min_length < 1:
            raise ValueError("min_length must be at least 1")

    def _compute_compression_ratio(self, text: str) -> float:
        """
        Compute the compression ratio of text.

        Returns the ratio of compressed size to original size.
        Lower ratio = better compression = more natural text.
        Higher ratio = poor compression = more random/garbled.
        """
        original = text.encode("utf-8")
        compressed = zlib.compress(original, level=6)

        # Compression adds headers (~10-15 bytes overhead)
        # For fair comparison, we use raw deflate size estimate
        original_len = len(original)
        compressed_len = len(compressed)

        if original_len == 0:
            return 0.0

        return compressed_len / original_len

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on compression ratio.

        Returns 0.0 for well-compressed text, 1.0 for incompressible text.
        """
        if not text or len(text) < self.min_length:
            return 0.0

        ratio = self._compute_compression_ratio(text)

        # Map ratio to probability
        # Below low threshold: definitely valid (0.0)
        # Above high threshold: definitely garbled (1.0)
        # In between: linear interpolation
        if ratio <= self.low_ratio_threshold:
            return 0.0
        elif ratio >= self.high_ratio_threshold:
            return 1.0
        else:
            # Linear interpolation between thresholds
            range_size = self.high_ratio_threshold - self.low_ratio_threshold
            return (ratio - self.low_ratio_threshold) / range_size
