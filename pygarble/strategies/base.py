import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict

# Pre-compiled regex for performance
_WHITESPACE_PATTERN = re.compile(r"\s")

# Long single tokens with these prefixes are common in real data (links,
# data URIs) and should not be auto-flagged by the long-string rule.
_URL_PREFIXES = ("http://", "https://", "ftp://", "file://", "data:", "www.")


class BaseStrategy(ABC):
    def __init__(self, **kwargs: Any):
        self.kwargs: Dict[str, Any] = kwargs

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
        max_length = self.kwargs.get("max_string_length", 1000)
        if len(text) <= max_length or _WHITESPACE_PATTERN.search(text):
            return False
        return not text.lower().startswith(_URL_PREFIXES)

    @staticmethod
    def _fold_diacritics(text: str) -> str:
        """Strip combining marks so ASCII n-gram models can score accented
        text (café -> cafe) instead of treating every accented n-gram as
        unseen."""
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(c for c in normalized if not unicodedata.combining(c))

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
        from ..data import ENGLISH_WORDS

        novel = []
        for token in text.split():
            if any(ch.isdigit() for ch in token):
                continue
            lower = token.lower()
            if "://" in lower or "@" in lower or lower.startswith("www."):
                continue
            alpha = "".join(
                c for c in self._fold_diacritics(token) if c.isalpha()
            )
            if not alpha:
                continue
            if token.isupper() and len(alpha) <= 6:
                continue
            if (
                skip_titlecase
                and token[:1].isupper()
                and token[1:].lower() == token[1:]
            ):
                continue
            alpha = alpha.lower()
            if alpha in ENGLISH_WORDS:
                continue
            novel.append(alpha)
        return novel

    def _predict_impl(self, text: str) -> bool:
        # Single source of truth: predict agrees with predict_proba unless a
        # strategy has a documented reason to override.
        return self._predict_proba_impl(text) >= 0.5

    @abstractmethod
    def _predict_proba_impl(self, text: str) -> float:
        pass
