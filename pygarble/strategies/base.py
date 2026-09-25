import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict

from ..analysis import Evidence
from ..options import validate_options
from ..preprocessing import TextFeatures, fold_diacritics
from ..validation import finite_number, positive_int

# Pre-compiled regex for performance
_WHITESPACE_PATTERN = re.compile(r"\s")

# Long single tokens with these prefixes are common in real data (links,
# data URIs) and should not be auto-flagged by the long-string rule.
_URL_PREFIXES = ("http://", "https://", "ftp://", "file://", "data:", "www.")


class BaseStrategy(ABC):
    def __init__(self, **kwargs: Any):
        self.kwargs: Dict[str, Any] = kwargs
        validate_options(type(self).__name__, kwargs)
        for name, value in kwargs.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                finite_number(name, value)
        if "max_string_length" in kwargs:
            positive_int("max_string_length", kwargs["max_string_length"])

    def predict(self, text: str) -> bool:
        self._validate_input(text)
        if not text or not text.strip():
            return False

        if self._is_extremely_long_string(text):
            return True

        return self._predict_impl(text)

    def predict_proba(self, text: str) -> float:
        self._validate_input(text)
        if not text or not text.strip():
            return 0.0

        if self._is_extremely_long_string(text):
            return 1.0

        return self._predict_proba_impl(text)

    def applicable(self, text: str) -> bool:
        """Whether this strategy can render a meaningful judgment on text.

        Strategies that need a minimum amount of text (e.g. word-level
        statistics) override this. EnsembleDetector only counts votes from
        applicable strategies.
        """
        return True

    @staticmethod
    def _validate_input(text: Any) -> None:
        if not isinstance(text, str):
            raise TypeError(
                f"text must be a string, got {type(text).__name__}"
            )

    def _is_extremely_long_string(self, text: str) -> bool:
        # Explicit compatibility option only. Length is not universal evidence.
        max_length = self.kwargs.get("max_string_length")
        if max_length is None:
            return False
        if len(text) <= max_length or _WHITESPACE_PATTERN.search(text):
            return False
        return not text.lower().startswith(_URL_PREFIXES)

    @staticmethod
    def _fold_diacritics(text: str) -> str:
        """Strip combining marks so ASCII n-gram models can score accented
        text (café -> cafe) instead of treating every accented n-gram as
        unseen."""
        return fold_diacritics(text)

    def _get_alpha_char_counts(self, text: str) -> Counter:
        return Counter(c for c in text.lower() if c.isalpha())

    def _novel_words(self, text: str, skip_titlecase: bool = False) -> list:
        """Lowercased alphabetic words that cannot be vouched for: not in
        the dictionary, not short acronyms, and free of digits/URL markers.

        Letter-pattern strategies (phonotactics, keyboard rows, character
        models) score only these, so real-but-rare words ("fjord",
        "rhythms"), acronyms (HTTP), and URLs don't register as gibberish
        while unknown tokens are still judged. skip_titlecase additionally
        drops likely proper nouns (Nguyen, McDonald) for strategies whose
        rules don't hold for names.
        """
        return [
            token.folded
            for token in TextFeatures(text).novel
            if not (skip_titlecase and token.text.istitle())
        ]

    def evaluate(self, features: TextFeatures) -> Evidence:
        self._validate_input(features.text)
        if not features.text.strip():
            return Evidence(0.0, False, "empty_input")
        if self._is_extremely_long_string(features.text):
            return Evidence(1.0, True, "explicit_length_policy")
        result = self._evaluate_features(features)
        score = finite_number("strategy score", result.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("strategy score must be between 0.0 and 1.0")
        return result

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        text = features.scrubbed
        if not text.strip():
            return Evidence(0.0, False, "insufficient_evidence")
        if not self.applicable(text):
            return Evidence(0.0, False, "insufficient_evidence")
        return Evidence(self._predict_proba_impl(text))

    def _predict_impl(self, text: str) -> bool:
        # Single source of truth: predict agrees with predict_proba unless a
        # strategy has a documented reason to override.
        return self._predict_proba_impl(text) >= 0.5

    @abstractmethod
    def _predict_proba_impl(self, text: str) -> float:
        pass
