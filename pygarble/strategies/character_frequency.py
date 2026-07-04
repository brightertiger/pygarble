from .base import BaseStrategy


class CharacterFrequencyStrategy(BaseStrategy):
    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_alpha_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        min_chars = self.kwargs.get("min_alpha_chars", 15)
        if total_chars < min_chars:
            # A max-frequency ratio over a handful of characters is noise
            # ("see", "noon") - unless the text is literally one repeated
            # character, which is unambiguous.
            if len(char_counts) == 1 and total_chars >= 5:
                return 1.0
            return 0.0

        max_frequency = max(char_counts.values()) / total_chars
        # Normal English peak is ~13% (letter 'e')
        # Only flag when a single character dominates
        threshold = self.kwargs.get("frequency_threshold", 0.2)
        if max_frequency <= threshold:
            return 0.0
        return min((max_frequency - threshold) / 0.8, 1.0)
