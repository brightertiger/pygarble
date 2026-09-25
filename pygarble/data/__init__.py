"""Lazy public exports; resources load only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bigrams import BIGRAM_LOG_PROBS as BIGRAM_LOG_PROBS
    from .bigrams import DEFAULT_LOG_PROB as DEFAULT_LOG_PROB
    from .function_words import FUNCTION_WORDS as FUNCTION_WORDS
    from .trigrams import COMMON_TRIGRAMS as COMMON_TRIGRAMS
    from .words import ENGLISH_WORDS as ENGLISH_WORDS

_EXPORTS = {
    "ENGLISH_WORDS": "words",
    "BIGRAM_LOG_PROBS": "bigrams",
    "DEFAULT_LOG_PROB": "bigrams",
    "COMMON_TRIGRAMS": "trigrams",
    "FUNCTION_WORDS": "function_words",
}
__all__ = [
    "ENGLISH_WORDS",
    "BIGRAM_LOG_PROBS",
    "DEFAULT_LOG_PROB",
    "COMMON_TRIGRAMS",
    "FUNCTION_WORDS",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
