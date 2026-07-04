import re

from .base import BaseStrategy

# A run of one repeated symbol ("----", "====") is formatting, not
# garble; collapse it to a single char before measuring symbol density.
_SYMBOL_RUN = re.compile(r"([^\w\s])\1{2,}")


class StatisticalAnalysisStrategy(BaseStrategy):
    def _get_symbol_ratio(self, text: str) -> float:
        text = _SYMBOL_RUN.sub(r"\1", text)
        content_chars = sum(1 for c in text if not c.isspace())
        if content_chars == 0:
            return 0.0

        # Digits and whitespace are neutral content: "2 + 2 = 4" or
        # "$100.50" is not garble. Only symbol characters count against
        # the text.
        symbol_chars = sum(
            1 for c in text if not c.isspace() and not c.isalnum()
        )
        return symbol_chars / content_chars

    def _predict_proba_impl(self, text: str) -> float:
        ratio = self._get_symbol_ratio(text)
        # Mild punctuation must not score: up to 30% symbols maps below
        # 0.5; only symbol-dominated text crosses the decision boundary.
        if ratio <= 0.3:
            return ratio / 0.3 * 0.4
        return 0.4 + (ratio - 0.3) / 0.7 * 0.6
