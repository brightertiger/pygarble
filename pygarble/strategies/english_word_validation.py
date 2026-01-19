import re
from typing import Any, List, Optional

from .base import BaseStrategy


class EnglishWordValidationStrategy(BaseStrategy):
    """
    Validate text using pyspellchecker dictionary.

    Note: This strategy requires the optional 'pyspellchecker' dependency.
    Install with: pip install pygarble[spellchecker]

    Consider using WORD_LOOKUP strategy instead, which has no external
    dependencies and uses an embedded 50K word dictionary.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._spell_checker: Optional[Any] = None

    @property
    def spell_checker(self) -> Any:
        """Lazy load spellchecker to defer import error."""
        if self._spell_checker is None:
            try:
                from spellchecker import SpellChecker
                self._spell_checker = SpellChecker()
            except ImportError:
                raise ImportError(
                    "pyspellchecker is required for EnglishWordValidationStrategy. "
                    "Install with: pip install pygarble[spellchecker]\n"
                    "Or use Strategy.WORD_LOOKUP which has no dependencies."
                )
        return self._spell_checker

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
