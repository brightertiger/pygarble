"""Deflate size of the text with an English word dictionary preset."""

from functools import lru_cache

from ..measures import primed_compression
from ._windowed import WindowedStrategy, reference_text


@lru_cache(maxsize=None)
def _dictionary() -> bytes:
    return reference_text().encode("ascii")


class PrimedCompressionStrategy(WindowedStrategy):
    """Compressed size per character with English words as a preset.

    The text is deflated with a preset dictionary of frequent English
    words. English finds long matches in the dictionary and compresses
    well; invented or mashed text does not. This is an English-reference
    method, so other languages are flagged too. It needs at least
    ``min_length`` letters (default 8) and grows more reliable with
    length.

    Compressed sizes come from the platform's zlib, so scores can differ
    slightly between zlib builds (for example zlib-ng).

    Args:
        midpoint: standardised value at which the score is 0.5
            (default 1.5; 1.0 is the null's 99th percentile)
        scale: sigmoid steepness, positive (default 2.0)
        min_length: minimum normalised characters to judge (default 8)

    Example:
        >>> detector = GarbleDetector(Strategy.PRIMED_COMPRESSION)
        >>> detector.predict("The meeting has been moved to Thursday.")
        False
    """

    statistic = "primed_compression"
    reason = "primed_compression_ratio"

    def _raw(self, window: str) -> float:
        return primed_compression(window, _dictionary())
