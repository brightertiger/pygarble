import re

from ..data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB
from .base import BaseStrategy

_WORD_PATTERN = re.compile(r"[a-z]+")


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.word_log_prob_threshold = kwargs.get(
            "word_log_prob_threshold", -4.6
        )
        self.min_word_length = kwargs.get("min_word_length", 4)
        self.anomaly_weight = kwargs.get("anomaly_weight", 2.0)

    def applicable(self, text: str) -> bool:
        return len(self._scoreable_words(text)) >= 1

    def _scoreable_words(self, text):
        folded = self._fold_diacritics(text).lower()
        return [
            w
            for w in _WORD_PATTERN.findall(folded)
            if len(w) >= self.min_word_length
        ]

    def _word_log_prob(self, word: str) -> float:
        padded = f" {word} "
        bigrams = [padded[i : i + 2] for i in range(len(padded) - 1)]
        total = sum(
            BIGRAM_LOG_PROBS.get(bg, DEFAULT_LOG_PROB) for bg in bigrams
        )
        return total / len(bigrams)

    def _predict_proba_impl(self, text: str) -> float:
        words = self._scoreable_words(text)
        if not words:
            return 0.0
        anomalous = sum(
            1
            for w in words
            if self._word_log_prob(w) < self.word_log_prob_threshold
        )
        fraction = anomalous / len(words)
        return min(1.0, fraction * self.anomaly_weight)
