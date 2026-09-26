"""
Markov Chain strategy for garble detection.

Uses character bigram transition probabilities trained on English text.
Garbled text will have low probability under the English language model.
"""

from typing import Any, Optional

from ...validation import finite_number, parameter_value
from ..analysis import Evidence
from ..preprocessing import TextFeatures
from ..scoring import bigram_stats, sigmoid
from .base import BaseStrategy


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
        considered garbled. Default is -3.5.
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
        self.threshold_per_char = finite_number(
            "threshold_per_char", kwargs.get("threshold_per_char", -3.5)
        )
        self.min_length: int = parameter_value(
            "min_length", kwargs.get("min_length", 3), 3
        )

        if self.threshold_per_char > 0:
            raise ValueError(
                (
                    "threshold_per_char must be non-positive (log probabilit"
                    "ies are negative)"
                )
            )

    def _mean(self, features: TextFeatures) -> Optional[float]:
        cleaned = " ".join(token.folded for token in features.novel)
        if len(cleaned) < self.min_length:
            return None
        # Preserve the historical single boundary between novel words.
        total, count = bigram_stats((cleaned,))
        return total / count

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        mean = self._mean(features)
        if mean is None:
            # Known words provide English evidence without novel-word scoring.
            return Evidence(0.0, bool(features.tokens), "known_or_short_words")
        reason = "unlikely_english_transitions"
        if any(not token.folded.isascii() for token in features.novel):
            reason = "outside_english_alphabet"
        return Evidence(
            sigmoid((self.threshold_per_char - mean) * 2.0), True, reason
        )

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
