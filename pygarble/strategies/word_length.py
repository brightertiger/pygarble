from typing import List

from .base import BaseStrategy


class WordLengthStrategy(BaseStrategy):
    def _get_words(self, text: str) -> List[str]:
        return text.split()

    def _predict_impl(self, text: str) -> bool:
        words = self._get_words(text)
        if not words:
            return False

        avg_length = sum(len(word) for word in words) / len(words)
        max_length: int = self.kwargs.get("max_word_length", 20)
        return avg_length > max_length

    def _predict_proba_impl(self, text: str) -> float:
        words = self._get_words(text)
        if not words:
            return 0.0

        avg_length = sum(len(word) for word in words) / len(words)
        max_length: int = self.kwargs.get("max_word_length", 20)

        if avg_length <= max_length:
            return 0.0

        return min((avg_length - max_length) / max_length, 1.0)
