"""Bounded token windows find severe corruption embedded in English prose."""

from typing import Any, List

from ..analysis import Evidence, Span
from ..preprocessing import TextFeatures
from ..scoring import word_log_probability
from ..validation import finite_number, positive_int
from .base import BaseStrategy


class LocalAnomalyStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_word_length = positive_int(
            "min_word_length", kwargs.get("min_word_length", 8)
        )
        self.word_log_prob_threshold = finite_number(
            "word_log_prob_threshold",
            kwargs.get("word_log_prob_threshold", -5.5),
        )
        self.window_words = positive_int(
            "window_words", kwargs.get("window_words", 4)
        )
        if self.window_words > 32:
            raise ValueError("window_words must be at most 32")
        if self.word_log_prob_threshold >= 0:
            raise ValueError("word_log_prob_threshold must be negative")

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        candidates = {
            (token.start, token.end): token
            for token in features.novel
            if token.folded.isascii()
        }
        spans: List[Span] = []
        flags = []
        for token in features.tokens:
            bad = (token.start, token.end) in candidates and (
                len(token.folded) >= 4
                and word_log_probability(token.folded)
                < self.word_log_prob_threshold
            )
            flags.append(int(bad))
            if bad and len(token.folded) >= self.min_word_length:
                spans.append(
                    Span(token.start, token.end, "severe_local_anomaly")
                )
        prefix = [0]
        for flag in flags:
            prefix.append(prefix[-1] + flag)
        for end in range(self.window_words, len(flags) + 1):
            start = end - self.window_words
            if prefix[end] - prefix[start] >= max(
                2, (self.window_words + 1) // 2
            ):
                spans.append(
                    Span(
                        features.tokens[start].start,
                        features.tokens[end - 1].end,
                        "corrupt_token_window",
                    )
                )
        return Evidence(
            0.8 if spans else 0.0,
            bool(candidates),
            "localized_english_anomaly",
            tuple(spans),
        )

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
