"""Pure statistics for the non-parametric strategies.

Every function takes its reference data as an argument and never imports
``pygarble.data``, so the data generator can calibrate them against freshly
built tables. Higher values always mean less like English.
"""

import zlib
from collections import Counter
from typing import List, Mapping, MutableSequence, Sequence, Tuple, TypeVar

T = TypeVar("T")

WINDOW_LENGTH = 127
MIN_TAIL_LENGTH = 64
BUCKET_EDGES = (16, 32, 64)

_MULTIPLIER = 6364136223846793005
_INCREMENT = 1442695040888963407
_MASK = 2**64 - 1


class Lcg:
    """64-bit linear congruential generator, stable across platforms."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def below(self, n: int) -> int:
        """Return an integer in ``[0, n)``, advancing the state first."""
        self.state = (self.state * _MULTIPLIER + _INCREMENT) & _MASK
        return ((self.state >> 11) * n) >> 53

    def shuffle(self, items: MutableSequence[T]) -> None:
        """Fisher-Yates shuffle in place."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def windows(s: str) -> List[str]:
    """Split long text greedily on spaces into windows of 127 characters.

    A word longer than a window is its own window. A final window shorter
    than 64 characters is dropped when there is more than one.
    """
    if len(s) <= WINDOW_LENGTH:
        return [s]
    result: List[str] = []
    current = ""
    for word in s.split(" "):
        if current and len(current) + 1 + len(word) <= WINDOW_LENGTH:
            current += " " + word
        else:
            if current:
                result.append(current)
            current = word
    result.append(current)
    if len(result) > 1 and len(result[-1]) < MIN_TAIL_LENGTH:
        result.pop()
    return result


def bucket(length: int) -> int:
    """Length bucket of a window: edges 16, 32 and 64."""
    for index, edge in enumerate(BUCKET_EDGES):
        if length < edge:
            return index
    return len(BUCKET_EDGES)


def standardised(
    raw: float, bucket: int, table: Sequence[Tuple[float, float]]
) -> float:
    """Scale ``raw`` so the null median is 0 and the null q99 is 1."""
    median, q99 = table[bucket]
    return (raw - median) / (q99 - median)


def cross_parsing(window: str, reference: str) -> float:
    """Ziv-Merhav cross parsing: phrases per character against a reference.

    Each phrase is the longest prefix of the rest of ``window`` found in
    ``reference``; an unseen character is a phrase of length one.
    """
    n = len(window)
    i = phrases = 0
    while i < n:
        k = 1
        p = 0
        while i + k <= n:
            # A longer prefix also contains the shorter one, so resume at p.
            q = reference.find(window[i : i + k], p)
            if q < 0:
                break
            p = q
            k += 1
        i += max(k - 1, 1)
        phrases += 1
    return phrases / n


def primed_compression(window: str, dictionary: bytes) -> float:
    """Raw deflate size per character with a preset dictionary."""
    compressor = zlib.compressobj(
        9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, dictionary
    )
    data = window.encode("ascii")
    compressed = compressor.compress(data) + compressor.flush()
    return len(compressed) / len(window)


def ngram_rank_distance(window: str, ranks: Mapping[str, int]) -> float:
    """Cavnar-Trenkle out-of-place distance over 1- to 3-grams."""
    padded = " " + window + " "
    counts: Counter = Counter()
    for n in (1, 2, 3):
        for i in range(len(padded) - n + 1):
            counts[padded[i : i + n]] += 1
    size = len(ranks)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    profile = [gram for gram, _ in ordered[:size]]
    distance = 0
    for position, gram in enumerate(profile):
        rank = ranks.get(gram)
        distance += size if rank is None else abs(position - rank)
    return distance / (size * len(profile))


def _mean_log_prob(
    text: str, log_probs: Mapping[str, float], default_log_prob: float
) -> float:
    total = 0.0
    for i in range(len(text) - 1):
        total += log_probs.get(text[i : i + 2], default_log_prob)
    return total / max(len(text) - 1, 1)


def permutation_gap(
    window: str,
    log_probs: Mapping[str, float],
    default_log_prob: float,
    shuffles: int = 8,
) -> float:
    """Mean bigram log likelihood of shuffles minus that of ``window``.

    Letters are shuffled among themselves and spaces stay in place, using an
    LCG seeded by the window's CRC-32, so the result is deterministic.
    """
    rng = Lcg(zlib.crc32(window.encode("ascii")))
    letters = [c for c in window if c != " "]
    total = 0.0
    for _ in range(shuffles):
        rng.shuffle(letters)
        chars = iter(letters)
        shuffled = "".join(c if c == " " else next(chars) for c in window)
        total += _mean_log_prob(shuffled, log_probs, default_log_prob)
    original = _mean_log_prob(window, log_probs, default_log_prob)
    return total / shuffles - original
