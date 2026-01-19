"""
Pre-computed data for pygarble detection strategies.

This module contains embedded lookup tables generated from
Peter Norvig's word frequency data (https://norvig.com/ngrams/).

All data is MIT licensed and can be freely used.
"""

from .words import ENGLISH_WORDS
from .bigrams import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB
from .trigrams import COMMON_TRIGRAMS

__all__ = [
    "ENGLISH_WORDS",
    "BIGRAM_LOG_PROBS",
    "DEFAULT_LOG_PROB",
    "COMMON_TRIGRAMS",
]
