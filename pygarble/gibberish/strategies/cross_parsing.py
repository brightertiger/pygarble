"""Ziv-Merhav cross parsing against an English word reference."""

from ..measures import cross_parsing
from ._windowed import WindowedStrategy, reference_text


class CrossParsingStrategy(WindowedStrategy):
    """Phrases needed to spell the text from common English words.

    The text is parsed left to right into the longest pieces found
    anywhere in a reference built from frequent English words. English
    reuses long pieces of the reference; invented or mashed text breaks
    into many short ones. This is an English-reference method: other
    languages written in Latin letters may be flagged, more so the less
    they resemble English, and text with no ASCII letters is not scored.
    It needs at least ``min_length`` letters (default 8) and grows more
    reliable with length.

    Args:
        midpoint: standardised value at which the score is 0.5
            (default 1.5; 1.0 is the null's 99th percentile)
        scale: sigmoid steepness, positive (default 2.0)
        min_length: minimum normalised characters to judge (default 8)

    Example:
        >>> detector = GarbleDetector(Strategy.CROSS_PARSING)
        >>> detector.predict("The meeting has been moved to Thursday.")
        False
    """

    statistic = "cross_parsing"
    reason = "cross_parsing_rate"

    def _raw(self, window: str) -> float:
        return cross_parsing(window, reference_text())
