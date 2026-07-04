import math
from collections import Counter
from typing import List

from .base import BaseStrategy
from ..data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB

# Expected character-level entropy of English text (~4.1 bits). Observed
# entropy is normalized against this (capped by the maximum entropy the
# text length allows) rather than the text's own alphabet size, so that
# any evenly-distributed text does not trivially score as "high entropy".
ENGLISH_CHAR_ENTROPY = 4.1

# Average per-bigram log-probabilities: typical English sits around -3;
# random keyboard mash drifts toward DEFAULT_LOG_PROB. Map this range
# onto [0, 1].
_BIGRAM_SCORE_FLOOR = -4.5
_BIGRAM_SCORE_RANGE = 3.0


class EntropyBasedStrategy(BaseStrategy):
    def _calculate_entropy(self, char_counts: Counter) -> float:
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy

    def _get_bigrams(self, text: str) -> List[str]:
        bigrams: List[str] = []
        for word in self._fold_diacritics(text).lower().split():
            alpha_word = "".join(c for c in word if c.isalpha())
            bigrams.extend(
                alpha_word[i:i+2] for i in range(len(alpha_word) - 1)
            )
        return bigrams

    def _get_bigram_score(self, text: str) -> float:
        bigrams = self._get_bigrams(text)
        if not bigrams:
            return 0.0

        avg_log_prob = sum(
            BIGRAM_LOG_PROBS.get(bg, DEFAULT_LOG_PROB) for bg in bigrams
        ) / len(bigrams)

        score = (-avg_log_prob + _BIGRAM_SCORE_FLOOR) / _BIGRAM_SCORE_RANGE
        return min(max(score, 0.0), 1.0)

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_alpha_char_counts(self._fold_diacritics(text))
        if not char_counts:
            return 0.0

        total_chars = sum(char_counts.values())
        entropy = self._calculate_entropy(char_counts)
        max_entropy = ENGLISH_CHAR_ENTROPY
        if total_chars > 1:
            max_entropy = min(ENGLISH_CHAR_ENTROPY, math.log2(total_chars))
        normalized_entropy = min(entropy / max_entropy, 1.0)
        entropy_score = 1.0 - normalized_entropy

        bigram_score = self._get_bigram_score(text)

        return entropy_score * 0.4 + bigram_score * 0.6
