from typing import List

from .base import BaseStrategy


class WordLengthStrategy(BaseStrategy):
    @staticmethod
    def _is_url_like(word: str) -> bool:
        """URLs and email addresses are legitimately long single tokens
        and must not drive the average word length."""
        lowered = word.lower()
        return "://" in lowered or "@" in lowered or lowered.startswith("www.")

    def _get_words(self, text: str) -> List[str]:
        return [w for w in text.split() if not self._is_url_like(w)]

    @staticmethod
    def _median_length(words: List[str]) -> float:
        lengths = sorted(len(word) for word in words)
        mid = len(lengths) // 2
        if len(lengths) % 2:
            return float(lengths[mid])
        return (lengths[mid - 1] + lengths[mid]) / 2.0

    def _predict_proba_impl(self, text: str) -> float:
        words = self._get_words(text)
        if not words:
            return 0.0

        # Median rather than mean: one long token must not dominate a
        # text of otherwise normal words.
        median_length = self._median_length(words)
        max_length: int = self.kwargs.get("max_word_length", 20)

        if median_length <= max_length:
            return 0.0

        return min((median_length - max_length) / max_length, 1.0)
