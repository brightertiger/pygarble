from .base import BaseStrategy


class CharacterFrequencyStrategy(BaseStrategy):
    def _predict_impl(self, text: str) -> bool:
        char_counts = self._get_alpha_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return False

        threshold = self.kwargs.get("frequency_threshold", 0.1)
        return any(count / total_chars > threshold for count in char_counts.values())

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_alpha_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        max_frequency = max(char_counts.values()) / total_chars
        return min(max_frequency, 1.0)
