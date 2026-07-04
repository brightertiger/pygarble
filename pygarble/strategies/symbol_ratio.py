"""
Symbol ratio strategy for garble detection.

Detects text with high proportion of special characters, numbers,
or non-alphabetic content.
"""

import re
from typing import Any

from .base import BaseStrategy


class SymbolRatioStrategy(BaseStrategy):
    """
    Detect garbled text based on symbol/number density.

    This strategy analyzes the ratio of non-alphabetic characters
    (symbols, numbers, punctuation) to total characters. Text with
    unusually high symbol density is flagged as garbled.

    This is particularly effective for:
    - Symbol spam (!!!@@@###)
    - Special character patterns (##$$%%^^)

    Parameters
    ----------
    symbol_threshold : float, optional
        Ratio of symbol characters above which text is considered
        garbled. Default is 0.5 (50% symbols).

    min_length : int, optional
        Minimum text length to analyze. Default is 3.

    allow_spaces : bool, optional
        If True, spaces are not counted as symbols. Default is True.

    count_digits : bool, optional
        If True, digits are counted as symbols, so numeric text
        (phone numbers, prices) is treated as garble. Default is
        False: digits count as normal characters.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.SYMBOL_RATIO)
    >>> detector.predict("!!!@@@###$$$")
    True
    >>> detector.predict("hello world")
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.symbol_threshold = kwargs.get("symbol_threshold", 0.5)
        self.min_length = kwargs.get("min_length", 3)
        self.allow_spaces = kwargs.get("allow_spaces", True)
        self.count_digits = kwargs.get("count_digits", False)

        if not 0.0 <= self.symbol_threshold <= 1.0:
            raise ValueError("symbol_threshold must be between 0.0 and 1.0")
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")

    def _compute_symbol_ratio(self, text: str) -> float:
        """
        Compute ratio of non-alphabetic characters.

        Returns a value between 0 and 1 where higher means more
        symbols/numbers (more likely garbled).
        """
        if not text:
            return 0.0

        # A run of one repeated symbol ("----", "====") is formatting,
        # not garble; collapse it before measuring density
        text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)

        # Count characters
        total = 0
        symbols = 0

        for c in text:
            # Skip spaces if configured
            if self.allow_spaces and c.isspace():
                continue

            total += 1
            if c.isalpha():
                continue
            if c.isdigit() and not self.count_digits:
                # Digits are normal characters unless configured otherwise
                continue
            symbols += 1

        if total < self.min_length or total == 0:
            return 0.0

        return symbols / total

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on symbol ratio.

        Returns a value between 0 and 1 where:
        - 0.0 = mostly alphanumeric (normal text)
        - 1.0 = mostly symbols (garbled)
        """
        symbol_ratio = self._compute_symbol_ratio(text)

        if symbol_ratio == 0.0:
            return 0.0

        # Map symbol ratio to garble score
        # Below threshold: scale 0 to 0.4
        # Above threshold: scale 0.5 to 1.0

        if symbol_ratio <= self.symbol_threshold:
            # Below threshold - low garble score
            if self.symbol_threshold > 0:
                normalized = symbol_ratio / self.symbol_threshold
                return 0.4 * normalized
            else:
                return 0.4  # threshold is 0, any symbol is suspicious
        else:
            # Above threshold - high garble score
            range_above = 1.0 - self.symbol_threshold
            if range_above > 0:
                normalized = (symbol_ratio - self.symbol_threshold) / range_above
                return 0.5 + 0.5 * normalized
            else:
                return 1.0
