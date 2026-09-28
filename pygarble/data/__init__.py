"""Lazy public exports; resources load only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bigrams import BIGRAM_LOG_PROBS as BIGRAM_LOG_PROBS
    from .bigrams import DEFAULT_LOG_PROB as DEFAULT_LOG_PROB
    from .calibration import SCORE_NULL_TAILS as SCORE_NULL_TAILS
    from .calibration import STATISTIC_NULL as STATISTIC_NULL
    from .calibration import TAIL_GRID as TAIL_GRID
    from .function_words import FUNCTION_WORDS as FUNCTION_WORDS
    from .ngram_ranks import NGRAM_RANKS as NGRAM_RANKS
    from .reference import REFERENCE_WORDS as REFERENCE_WORDS
    from .trigrams import COMMON_TRIGRAMS as COMMON_TRIGRAMS
    from .words import ENGLISH_WORDS as ENGLISH_WORDS

_EXPORTS = {
    "ENGLISH_WORDS": "words",
    "BIGRAM_LOG_PROBS": "bigrams",
    "DEFAULT_LOG_PROB": "bigrams",
    "COMMON_TRIGRAMS": "trigrams",
    "FUNCTION_WORDS": "function_words",
    "REFERENCE_WORDS": "reference",
    "NGRAM_RANKS": "ngram_ranks",
    "STATISTIC_NULL": "calibration",
    "SCORE_NULL_TAILS": "calibration",
    "TAIL_GRID": "calibration",
}
__all__ = [
    "ENGLISH_WORDS",
    "BIGRAM_LOG_PROBS",
    "DEFAULT_LOG_PROB",
    "COMMON_TRIGRAMS",
    "FUNCTION_WORDS",
    "REFERENCE_WORDS",
    "NGRAM_RANKS",
    "STATISTIC_NULL",
    "SCORE_NULL_TAILS",
    "TAIL_GRID",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
