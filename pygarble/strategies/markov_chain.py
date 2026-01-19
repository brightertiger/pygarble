"""
Markov Chain strategy for garble detection.

Uses character bigram transition probabilities trained on English text.
Garbled text will have low probability under the English language model.
"""

import math
from typing import Any

from .base import BaseStrategy
from ..data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB


class MarkovChainStrategy(BaseStrategy):
    """
    Detect garbled text using a character-level Markov chain.

    This strategy computes the log-probability of text under a bigram
    language model trained on English. Text with low probability
    (many unusual character transitions) is flagged as garbled.

    Parameters
    ----------
    threshold_per_char : float, optional
        Average log probability per character below which text is
        considered garbled. Default is -5.0.
        More negative = more permissive (accepts more text as valid)
        Less negative = more strict (flags more text as garbled)

    min_length : int, optional
        Minimum text length to analyze. Shorter texts return 0.0.
        Default is 3.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.MARKOV_CHAIN)
    >>> detector.predict("hello world")
    False
    >>> detector.predict("asdfghjkl")
    True
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Threshold tuned based on analysis:
        # - Valid English text: typically -2.0 to -3.0
        # - Keyboard mashing: typically -4.0 to -5.0
        # - Random gibberish: typically -5.0 to -8.0
        self.threshold_per_char = kwargs.get("threshold_per_char", -3.5)
        self.min_length = kwargs.get("min_length", 3)

        if self.threshold_per_char > 0:
            raise ValueError("threshold_per_char must be non-positive (log probabilities are negative)")
        if self.min_length < 1:
            raise ValueError("min_length must be at least 1")

    def _compute_log_probability(self, text: str) -> float:
        """
        Compute average log probability per character transition.

        Returns the average log probability of all bigrams in the text.
        Higher (less negative) values indicate more English-like text.
        """
        # Normalize: lowercase and add word boundaries
        text = text.lower()

        # Extract only alphabetic characters and spaces
        cleaned = "".join(c if c.isalpha() or c.isspace() else " " for c in text)
        # Collapse multiple spaces
        cleaned = " ".join(cleaned.split())

        if len(cleaned) < self.min_length:
            return 0.0

        # Add start/end markers
        padded = " " + cleaned + " "

        # Sum log probabilities of all bigrams
        total_log_prob = 0.0
        num_bigrams = 0

        for i in range(len(padded) - 1):
            bigram = padded[i:i + 2]
            log_prob = BIGRAM_LOG_PROBS.get(bigram, DEFAULT_LOG_PROB)
            total_log_prob += log_prob
            num_bigrams += 1

        if num_bigrams == 0:
            return 0.0

        # Return average log probability per bigram
        return total_log_prob / num_bigrams

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on Markov chain model.

        Maps log probability to a [0, 1] garble score where:
        - 0.0 = definitely valid English
        - 1.0 = definitely garbled
        """
        avg_log_prob = self._compute_log_probability(text)

        if avg_log_prob == 0.0:
            return 0.0

        # Map log probability to garble score
        # avg_log_prob typically ranges from about -2 (common text) to -8 (gibberish)
        # We use a sigmoid-like mapping centered on the threshold

        # Difference from threshold (positive = more garbled)
        diff = self.threshold_per_char - avg_log_prob

        # Scale to reasonable range and apply sigmoid
        # Factor of 2 gives good separation between valid/invalid
        scaled = diff * 2.0

        # Sigmoid function: 1 / (1 + exp(-x))
        try:
            garble_score = 1.0 / (1.0 + math.exp(-scaled))
        except OverflowError:
            garble_score = 0.0 if scaled < 0 else 1.0

        return min(1.0, max(0.0, garble_score))
