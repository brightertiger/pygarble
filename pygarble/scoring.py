"""Shared deterministic character likelihood primitives."""

import math
from typing import Iterable, Tuple


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


def bigram_stats(words: Iterable[str]) -> Tuple[float, int]:
    from .data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB

    total = 0.0
    count = 0
    for word in words:
        padded = " " + word + " "
        for i in range(len(padded) - 1):
            total += BIGRAM_LOG_PROBS.get(padded[i : i + 2], DEFAULT_LOG_PROB)
            count += 1
    return total, count


def word_log_probability(word: str) -> float:
    total, count = bigram_stats((word,))
    return total / count
