"""Frozen adapters and locally trained, inexpensive character models."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# The bigram approach is adapted from Rob Renaud's Gibberish-Detector.
# See LICENSES.md for its MIT notice and protocol.md for differences.
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
DEFAULTS = (
    "keep_all",
    "english",
    "english_extended",
    "legacy",
    "word_lookup",
    "entropy_based",
)
CALIBRATED = DEFAULTS[1:] + ("char_bigram", "char_trigram")
FOLDS = (
    ("bible", "wiki", "secreta"),
    ("secreta", "bible", "wiki"),
    ("wiki", "secreta", "bible"),
)


class CharacterModel:
    """Conditional character NLL; higher means less like training prose."""

    def __init__(self, order: int, texts: List[str]) -> None:
        self.order = order
        self.alpha = 10 if order == 2 else 1
        self.counts: Dict[str, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()
        for text in texts:
            value = self.normalize(text)
            for i in range(order - 1, len(value)):
                context = value[i - order + 1 : i]
                self.counts[context][value[i]] += 1
                self.totals[context] += 1

    @staticmethod
    def normalize(text: str) -> str:
        return "".join(c.lower() for c in text if c.lower() in ALPHABET)

    def score(self, text: str) -> Tuple[float, bool, str]:
        value = self.normalize(text)
        terms = []
        for i in range(self.order - 1, len(value)):
            context = value[i - self.order + 1 : i]
            count = self.counts.get(context, {}).get(value[i], 0)
            total = self.totals.get(context, 0)
            terms.append(
                -math.log(
                    (count + self.alpha) / (total + self.alpha * len(ALPHABET))
                )
            )
        if not terms:
            return 0.0, False, "insufficient_evidence"
        return sum(terms) / len(terms), True, "scored"


class PackageModel:
    def __init__(self, name: str) -> None:
        from pygarble import EnsembleDetector, GarbleDetector

        if name in ("english", "english_extended", "legacy"):
            self.detector = EnsembleDetector(profile=name, threads=1)
        else:
            self.detector = GarbleDetector(name, threads=1)

    def score(self, text: str) -> Tuple[float, bool, str]:
        analysis = self.detector.analyze(text)
        return (
            analysis.score,
            analysis.status != "insufficient_evidence",
            analysis.status,
        )


class KeepAll:
    def score(self, text: str) -> Tuple[float, bool, str]:
        return 0.0, True, "clean"


def make_model(name: str, training: List[str]) -> Any:
    if name == "keep_all":
        return KeepAll()
    if name.startswith("char_"):
        return CharacterModel(2 if name == "char_bigram" else 3, training)
    return PackageModel(name)


def next_float(value: float) -> float:
    """Next float above a finite nonnegative score, including on Python 3.8."""
    import struct

    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative score")
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits + 1))[0]


def threshold(
    scores: List[Tuple[float, bool, str]], budget: float = 0.01
) -> float:
    if not scores:
        raise ValueError("Calibration set is empty")
    if not 0 <= budget <= 1:
        raise ValueError("Invalid false-positive budget")
    active = [value for value, applicable, _ in scores if applicable]
    candidates = sorted({0.0} | {next_float(s) for s in active})
    for candidate in candidates:
        if sum(s >= candidate for s in active) / len(scores) <= budget:
            return candidate
    raise AssertionError("Keep-all candidate must satisfy the budget")
