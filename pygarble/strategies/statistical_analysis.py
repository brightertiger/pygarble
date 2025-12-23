from typing import Any

from .base import BaseStrategy


class StatisticalAnalysisStrategy(BaseStrategy):
    def _get_alpha_ratio(self, text: str) -> float:
        alpha_chars = sum(1 for c in text if c.isalpha())
        content_chars = sum(1 for c in text if not c.isspace())

        if content_chars == 0:
            return 0.0

        return alpha_chars / content_chars

    def _predict_impl(self, text: str) -> bool:
        alpha_ratio = self._get_alpha_ratio(text)
        threshold: float = self.kwargs.get("alpha_threshold", 0.5)
        return alpha_ratio < threshold

    def _predict_proba_impl(self, text: str) -> float:
        alpha_ratio = self._get_alpha_ratio(text)
        return 1.0 - alpha_ratio
