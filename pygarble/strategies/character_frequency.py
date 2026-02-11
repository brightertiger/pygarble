from .base import BaseStrategy


class CharacterFrequencyStrategy(BaseStrategy):
    def _predict_impl(self, text: str) -> bool:
        char_counts = self._get_alpha_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return False

        threshold = self.kwargs.get("frequency_threshold", 0.3)
        max_frequency = max(char_counts.values()) / total_chars
        return max_frequency > threshold

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_alpha_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        max_frequency = max(char_counts.values()) / total_chars
        # Normal English peak is ~13% (letter 'e')
        # Only flag when a single character dominates (>20%)
        if max_frequency <= 0.2:
            return 0.0
        return min((max_frequency - 0.2) / 0.8, 1.0)
