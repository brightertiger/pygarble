import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict


class BaseStrategy(ABC):
    def __init__(self, **kwargs: Any):
        self.kwargs: Dict[str, Any] = kwargs

    def predict(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        if self._is_extremely_long_string(text):
            return True

        return self._predict_impl(text)

    def predict_proba(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        if self._is_extremely_long_string(text):
            return 1.0

        return self._predict_proba_impl(text)

    def _is_extremely_long_string(self, text: str) -> bool:
        max_length = self.kwargs.get("max_string_length", 1000)
        return len(text) > max_length and not re.search(r"\s", text)

    def _get_alpha_char_counts(self, text: str) -> Counter:
        return Counter(c for c in text.lower() if c.isalpha())

    @abstractmethod
    def _predict_impl(self, text: str) -> bool:
        pass

    @abstractmethod
    def _predict_proba_impl(self, text: str) -> float:
        pass
