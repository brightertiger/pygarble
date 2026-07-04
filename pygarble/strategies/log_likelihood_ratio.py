import math
import re

from ..data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB
from .base import BaseStrategy

_WORD_PATTERN = re.compile(r"[a-z]+")

# Null hypothesis: characters drawn uniformly from a-z plus word boundary.
_UNIFORM_LOG_PROB = math.log(1.0 / 27.0)


class LogLikelihoodRatioStrategy(BaseStrategy):
    """Two-model character bigram log-likelihood ratio.

    Scores text by comparing how likely its character bigrams are under an
    English model versus a uniform "random typing" model:

        LLR = log P(text | english) - log P(text | uniform)

    English text averages a strongly positive LLR per bigram; random
    characters average a strongly negative one. Unlike absolute-probability
    thresholds this is self-normalizing for length, so short and long texts
    are scored on the same scale.

    Args:
        llr_midpoint: average per-bigram LLR mapped to probability 0.5
            (default -1.0; calibrated so rare-but-real words like
            "rhythms" at -0.6 stay clean while gibberish, which scores
            below -1.4, is flagged)
        llr_scale: steepness of the logistic mapping (default 1.5)
        min_bigrams: minimum bigram count needed to judge (default 3)

    Example:
        >>> detector = GarbleDetector(Strategy.LOG_LIKELIHOOD_RATIO)
        >>> detector.predict("the quick brown fox")
        False
        >>> detector.predict("xkjq zvwp qmfgh")
        True
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llr_midpoint = kwargs.get("llr_midpoint", -1.0)
        self.llr_scale = kwargs.get("llr_scale", 1.5)
        self.min_bigrams = kwargs.get("min_bigrams", 3)

    def applicable(self, text: str) -> bool:
        return len(self._extract_bigrams(text)) >= self.min_bigrams

    def _extract_bigrams(self, text):
        folded = self._fold_diacritics(text).lower()
        bigrams = []
        for word in _WORD_PATTERN.findall(folded):
            padded = f" {word} "
            bigrams.extend(
                padded[i : i + 2] for i in range(len(padded) - 1)
            )
        return bigrams

    def _average_llr(self, text: str) -> float:
        bigrams = self._extract_bigrams(text)
        if len(bigrams) < self.min_bigrams:
            return 0.0
        total = sum(
            BIGRAM_LOG_PROBS.get(bg, DEFAULT_LOG_PROB) - _UNIFORM_LOG_PROB
            for bg in bigrams
        )
        return total / len(bigrams)

    def _predict_proba_impl(self, text: str) -> float:
        bigrams = self._extract_bigrams(text)
        if len(bigrams) < self.min_bigrams:
            return 0.0
        avg_llr = self._average_llr(text)
        # Logistic map: LLR above midpoint -> clean, below -> garbled
        z = self.llr_scale * (avg_llr - self.llr_midpoint)
        # Clamp to avoid overflow in exp
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(z))
