from typing import Any, List

from ..analysis import Evidence, Span
from ..preprocessing import TextFeatures
from ..scoring import word_log_probability
from ..validation import finite_number, positive_int
from .base import BaseStrategy


class WordAnomalyStrategy(BaseStrategy):
    """Per-word anomaly scoring: flag text by its fraction of garbled tokens.

    Text-level averaging dilutes a single garbage token inside an otherwise
    valid sentence ("order confirmed asdkjfhq thanks"). This strategy scores
    each word independently against the English character bigram model and
    reports the fraction of anomalous words, so one clearly-mashed token in
    a short sentence still registers.

    Args:
        word_log_prob_threshold: average per-bigram log-probability below
            which a word is considered anomalous (default -4.6; English
            words typically average -2 to -3, random letters -6 to -10)
        min_word_length: only score words with at least this many letters
            (default 4 - short tokens have too few bigrams to judge)
        anomaly_weight: multiplier mapping the anomalous fraction to a
            probability (default 2.0, so 1 bad word out of 4 crosses 0.5)

    Example:
        >>> detector = GarbleDetector(Strategy.WORD_ANOMALY)
        >>> detector.predict("order confirmed asdkjfhq thanks")
        True
        >>> detector.predict("order confirmed successfully thanks")
        False
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.word_log_prob_threshold = finite_number(
            "word_log_prob_threshold",
            kwargs.get("word_log_prob_threshold", -4.6),
        )
        self.min_word_length = positive_int(
            "min_word_length", kwargs.get("min_word_length", 4)
        )
        self.anomaly_weight = finite_number(
            "anomaly_weight", kwargs.get("anomaly_weight", 2.0)
        )
        if self.anomaly_weight <= 0:
            raise ValueError("anomaly_weight must be positive")

    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return bool(self._scoreable_words(text))

    def _scoreable_words(self, text: str) -> List[str]:
        return [
            w
            for w in TextFeatures(text).ascii_words
            if len(w) >= self.min_word_length
        ]

    def _word_log_prob(self, word: str) -> float:
        return word_log_probability(word)

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        words = [
            w for w in features.ascii_words if len(w) >= self.min_word_length
        ]
        if not words:
            return Evidence(0.0, False, "insufficient_words")
        bad = {
            w
            for w in words
            if w not in features.allowlist
            and self._word_log_prob(w) < self.word_log_prob_threshold
        }
        score = min(
            1.0,
            sum(w in bad for w in words) / len(words) * self.anomaly_weight,
        )
        spans = tuple(
            Span(t.start, t.end, "anomalous_word")
            for t in features.tokens
            if t.folded in bad
        )
        return Evidence(score, True, "anomalous_word_fraction", spans)

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
