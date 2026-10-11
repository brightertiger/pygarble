"""Benchmark methods: a continuous score (higher = more junk) and a decision.

Each method is built lazily so pygarble-only runs do not need baselines.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

from .data import DEFAULT_ROOT

Scores = Tuple[List[float], List[bool]]
BASELINE_FILES = {
    "big.txt": "e77a1d4aeefb1f2c32fe0fc30551478a"
    "7eb59dc093f270c095822290971d39d7",
}
BASELINE_URLS = {
    "big.txt": "https://raw.githubusercontent.com/domanchi/"
    "gibberish-detector/ecac96918e72fbf702cfee3274dbf280009171ad/"
    "examples/big.txt",
}
GIBBERISH_DETECTOR_LIMIT = 4.0


class Method:
    """Scores texts; `decide` is the native call timed for footprint."""

    def run(self, texts: Sequence[str]) -> Scores:
        raise NotImplementedError

    def decide(self, texts: Sequence[str]) -> List[bool]:
        return self.run(texts)[1]


class Pygarble(Method):
    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def run(self, texts: Sequence[str]) -> Scores:
        texts = list(texts)
        scores = [float(s) for s in self.detector.predict_proba(texts)]
        return scores, [bool(d) for d in self.detector.predict(texts)]

    def decide(self, texts: Sequence[str]) -> List[bool]:
        return list(self.detector.predict(list(texts)))


class PerText(Method):
    """Baseline scored one text at a time."""

    def one(self, text: str) -> Tuple[float, bool]:
        raise NotImplementedError

    def run(self, texts: Sequence[str]) -> Scores:
        pairs = [self.one(text) for text in texts]
        return [s for s, _ in pairs], [d for _, d in pairs]


class GibberishDetector(PerText):
    def __init__(self, corpus: Path) -> None:
        from gibberish_detector import detector, trainer

        model = trainer.train(str(corpus))
        self.detector = detector.Detector(model, GIBBERISH_DETECTOR_LIMIT)

    def one(self, text: str) -> Tuple[float, bool]:
        d = self.detector
        score = float(d.calculate_probability_of_being_gibberish(text))
        return score, score > d.limit

    def decide(self, texts: Sequence[str]) -> List[bool]:
        return [self.detector.is_gibberish(text) for text in texts]


class Lingua(PerText):
    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self.english = Language.ENGLISH
        self.detector = LanguageDetectorBuilder.from_all_languages().build()

    def one(self, text: str) -> Tuple[float, bool]:
        confidence = self.detector.compute_language_confidence(
            text, self.english
        )
        detected = self.detector.detect_language_of(text)
        return 1.0 - float(confidence), detected != self.english

    def decide(self, texts: Sequence[str]) -> List[bool]:
        return [
            self.detector.detect_language_of(text) != self.english
            for text in texts
        ]


class Spec(NamedTuple):
    name: str
    family: str
    build: Callable[[], Method]
    imports: str  # statement timed for import cost
    distributions: Tuple[str, ...]


def _ensemble(**kwargs: Any) -> Callable[[], Method]:
    def build() -> Method:
        from pygarble import EnsembleDetector

        return Pygarble(EnsembleDetector(threads=1, **kwargs))

    return build


def specs(data_root: Path = DEFAULT_ROOT) -> Dict[str, Spec]:
    files = data_root / "baselines"
    ensemble = "from pygarble import EnsembleDetector"
    return {
        spec.name: spec
        for spec in (
            Spec(
                "english",
                "pygarble",
                _ensemble(profile="english"),
                ensemble,
                ("pygarble",),
            ),
            Spec(
                "english_fusion",
                "pygarble",
                _ensemble(profile="english_fusion"),
                ensemble,
                ("pygarble",),
            ),
            Spec(
                "gibberish_detector",
                "baseline",
                lambda: GibberishDetector(files / "big.txt"),
                "from gibberish_detector import detector, trainer",
                ("gibberish-detector",),
            ),
            Spec(
                "lingua",
                "baseline",
                Lingua,
                "from lingua import Language, LanguageDetectorBuilder",
                ("lingua-language-detector",),
            ),
        )
    }
