"""English-versus-uniform bigram likelihood ratio (a heuristic score)."""

import math
from typing import Any, List

from ..analysis import Evidence
from ..preprocessing import TextFeatures
from ..scoring import sigmoid
from ..validation import finite_number, positive_int
from .base import BaseStrategy


class LogLikelihoodRatioStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llr_midpoint = finite_number(
            "llr_midpoint", kwargs.get("llr_midpoint", -1.0)
        )
        self.llr_scale = finite_number(
            "llr_scale", kwargs.get("llr_scale", 1.5)
        )
        self.min_bigrams = positive_int(
            "min_bigrams", kwargs.get("min_bigrams", 3)
        )
        if self.llr_scale <= 0:
            raise ValueError("llr_scale must be positive")

    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return TextFeatures(text).bigram_stats[1] >= self.min_bigrams

    def _extract_bigrams(self, text: str) -> List[str]:
        return [
            padded[i : i + 2]
            for word in TextFeatures(text).ascii_words
            for padded in (" " + word + " ",)
            for i in range(len(padded) - 1)
        ]

    def _average_llr(self, text: str) -> float:
        total, count = TextFeatures(text).bigram_stats
        return (
            total / count + math.log(27.0)
            if count >= self.min_bigrams
            else 0.0
        )

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        total, count = features.bigram_stats
        if count < self.min_bigrams:
            return Evidence(0.0, False, "insufficient_bigrams")
        mean = total / count + math.log(27.0)
        return Evidence(
            sigmoid(self.llr_scale * (self.llr_midpoint - mean)),
            True,
            "english_uniform_likelihood_ratio",
        )

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
