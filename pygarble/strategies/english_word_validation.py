import re
from typing import Any, List

from spellchecker import SpellChecker

from .base import BaseStrategy


class EnglishWordValidationStrategy(BaseStrategy):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.spell_checker = SpellChecker()

    def _tokenize_text(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize_text(text)
        if not words:
            return 0.0

        unknown_words = self.spell_checker.unknown(words)
        return len(unknown_words) / len(words)
