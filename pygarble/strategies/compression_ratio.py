"""
Compression ratio strategy for garble detection.

Uses zlib compression to detect degenerate text: highly repetitive
spam (compresses extremely well) or binary-like/incompressible data
(barely compresses at all).
"""

import zlib
from typing import Any

from .base import BaseStrategy


class CompressionRatioStrategy(BaseStrategy):
    """
    Detect degenerate text based on zlib compression ratio.

    Important limitation: compression CANNOT separate random *letters*
    from normal English -- both compress to a similar ratio (~0.75)
    at typical lengths. What the compression ratio does reliably
    detect:

    - Highly repetitive spam ("abc abc abc ...") compresses extremely
      well -> very LOW ratio -> flagged.
    - Binary-ish/random-byte data barely compresses -> ratio near or
      above 1.0 -> flagged.
    - Normal prose (and random letters) land in the middle -> 0.0.

    Uses Python's built-in zlib (no external deps). Texts shorter than
    ``min_length`` cannot be judged (zlib header overhead dominates),
    so this strategy abstains on them (``applicable()`` is False).

    Parameters
    ----------
    low_ratio_threshold : float, optional
        Compression ratio at or below which text is considered
        degenerate (highly repetitive). Default is 0.4.

    high_ratio_threshold : float, optional
        Compression ratio at or above which text is considered
        binary-like / incompressible. Default is 0.95.

    min_length : int, optional
        Minimum text length to analyze. Short texts don't compress
        well regardless of content due to header overhead.
        Default is 100.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.COMPRESSION_RATIO)
    >>> detector.predict("abc " * 50)  # highly repetitive spam
    True
    >>> detector.predict(
    ...     "The quick brown fox jumps over the lazy dog while the "
    ...     "birds sing in the morning near the old wooden fence."
    ... )
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.high_ratio_threshold = kwargs.get("high_ratio_threshold", 0.95)
        self.low_ratio_threshold = kwargs.get("low_ratio_threshold", 0.4)
        self.min_length = kwargs.get("min_length", 100)

        if not 0.0 <= self.high_ratio_threshold <= 1.5:
            raise ValueError("high_ratio_threshold must be between 0.0 and 1.5")
        if not 0.0 <= self.low_ratio_threshold <= 1.5:
            raise ValueError("low_ratio_threshold must be between 0.0 and 1.5")
        if self.low_ratio_threshold >= self.high_ratio_threshold:
            raise ValueError("low_ratio_threshold must be less than high_ratio_threshold")
        if self.min_length < 1:
            raise ValueError("min_length must be at least 1")

    def applicable(self, text: str) -> bool:
        """Compression is meaningless below min_length (header overhead)."""
        return len(text) >= self.min_length

    def _compute_compression_ratio(self, text: str) -> float:
        """
        Compute the compression ratio of text.

        Returns the ratio of compressed size to original size.
        Very low ratio = highly repetitive.
        Ratio near/above 1.0 = incompressible (binary-like).
        """
        original = text.encode("utf-8")
        compressed = zlib.compress(original, level=6)

        original_len = len(original)
        compressed_len = len(compressed)

        if original_len == 0:
            return 0.0

        return compressed_len / original_len

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on compression ratio.

        LOW ratio (<= low_ratio_threshold): highly repetitive -> high score.
        HIGH ratio (>= high_ratio_threshold): binary-like -> high score.
        Mid-range (normal prose, but also random letters): 0.0.
        """
        if not text or len(text) < self.min_length:
            return 0.0

        ratio = self._compute_compression_ratio(text)

        if ratio <= self.low_ratio_threshold:
            # 0.5 at the threshold, approaching 1.0 as ratio -> 0
            depth = (self.low_ratio_threshold - ratio) / self.low_ratio_threshold
            return min(1.0, 0.5 + 0.5 * depth)

        if ratio >= self.high_ratio_threshold:
            # 0.5 at the threshold, 1.0 once ratio exceeds it by 0.2
            excess = ratio - self.high_ratio_threshold
            return min(1.0, 0.5 + excess / 0.2)

        return 0.0
