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
    "ldnoobw_en.txt": "af851ecef1d5f212caba17339b12ac39"
    "cc2fef7d78c74876f67237644fcee8bd",
}
BASELINE_URLS = {
    "big.txt": "https://raw.githubusercontent.com/domanchi/"
    "gibberish-detector/ecac96918e72fbf702cfee3274dbf280009171ad/"
    "examples/big.txt",
    "ldnoobw_en.txt": "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    "5faf2ba42d7b1c0977169ec3611df25a3c08eb13/en",
}
NOSTRIL_REJECTED = 1_000_000.0
NOSTRIL_MIN_SCORE = 8.2
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


class Nostril(PerText):
    def __init__(self) -> None:
        import nostril
        from nostril import nonsense_detector as nd

        self.nd = nd
        self.nonsense = nostril.nonsense
        # Same parameters as nostril's module-level `nonsense` detector.
        self.score = nd._tfidf_score_function(
            nd.dataset_from_pickle(
                str(Path(nd.__file__).with_name("ngram_data.pklz"))
            ),
            len_threshold=25,
            len_penalty_exp=0.9233,
            repetition_penalty_exp=0.9674,
        )

    def one(self, text: str) -> Tuple[float, bool]:
        try:
            decision = bool(self.nonsense(text))
        except ValueError:  # "Text is too short to test"
            return 0.0, False
        clean = self.nd.sanitize_string(text)
        if self.nd._simple_real(clean):
            return 0.0, decision
        if self.nd._simple_nonsense(clean):
            return NOSTRIL_REJECTED, decision
        score = float(self.score(clean))
        assert decision == (score > NOSTRIL_MIN_SCORE)
        return score, decision

    def decide(self, texts: Sequence[str]) -> List[bool]:
        out = []
        for text in texts:
            try:
                out.append(bool(self.nonsense(text)))
            except ValueError:
                out.append(False)
        return out


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


class GibberishChatDetector(PerText):
    def __init__(self) -> None:
        from gibberish_detector import gibberish

        self.module = gibberish

    def one(self, text: str) -> Tuple[float, bool]:
        features = self.module.gibberish_detector(text)
        # The package's own rule string on the package's own features.
        fired = eval(  # noqa: S307 - fixed rule text shipped by the package
            self.module.default_gibberish_rule, {}, dict(features)
        )
        return float(sum(map(bool, fired))), bool(features["gibberish"])

    def decide(self, texts: Sequence[str]) -> List[bool]:
        return [
            bool(self.module.gibberish_detector(text)["gibberish"])
            for text in texts
        ]


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


class Rules(PerText):
    def __init__(self, check: Callable[[str], List[str]]) -> None:
        self.check = check

    def one(self, text: str) -> Tuple[float, bool]:
        failed = len(self.check(text))
        return float(failed), failed > 0


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


def _strategy(name: str) -> Callable[[], Method]:
    def build() -> Method:
        from pygarble import GarbleDetector

        return Pygarble(GarbleDetector(name, threads=1))

    return build


def _gopher() -> Method:
    from . import rules

    return Rules(rules.gopher_failures)


def _c4(bad_words: Path) -> Callable[[], Method]:
    def build() -> Method:
        from . import rules

        words = rules.load_bad_words(bad_words)
        return Rules(lambda text: rules.c4_failures(text, words))

    return build


PROFILES = (
    "english",
    "english_extended",
    "english_fusion",
    "legacy",
    "corruption",
    "spoofing",
    "llm_output",
)
VOTING = ("majority", "any", "all", "average", "weighted", "fisher")
ENGLISH_MEMBERS = 6
BASELINES = (
    "nostril",
    "gibberish_detector",
    "gibberish_chat_detector",
    "lingua",
    "gopher_rules",
    "c4_rules",
)


def specs(data_root: Path = DEFAULT_ROOT) -> Dict[str, Spec]:
    from pygarble.gibberish.registry import Strategy

    files = data_root / "baselines"
    ensemble = "from pygarble import EnsembleDetector"
    out: List[Spec] = []
    for profile in PROFILES:
        out.append(
            Spec(
                profile,
                "profile",
                _ensemble(profile=profile),
                ensemble,
                ("pygarble",),
            )
        )
    for mode in VOTING:
        weights = [1.0] * ENGLISH_MEMBERS if mode == "weighted" else None
        out.append(
            Spec(
                f"english_vote_{mode}",
                "voting",
                _ensemble(profile="english", voting=mode, weights=weights),
                ensemble,
                ("pygarble",),
            )
        )
    for strategy in Strategy:
        out.append(
            Spec(
                f"strategy_{strategy.value}",
                "strategy",
                _strategy(strategy.value),
                "from pygarble import GarbleDetector",
                ("pygarble",),
            )
        )
    out += [
        Spec(
            "nostril",
            "baseline",
            Nostril,
            "import nostril",
            ("nostril",),
        ),
        Spec(
            "gibberish_detector",
            "baseline",
            lambda: GibberishDetector(files / "big.txt"),
            "from gibberish_detector import detector, trainer",
            ("gibberish-detector",),
        ),
        Spec(
            "gibberish_chat_detector",
            "baseline",
            GibberishChatDetector,
            "from gibberish_detector import gibberish",
            ("gibberish-chat-detector",),
        ),
        Spec(
            "lingua",
            "baseline",
            Lingua,
            "from lingua import Language, LanguageDetectorBuilder",
            ("lingua-language-detector",),
        ),
        Spec(
            "gopher_rules",
            "baseline",
            _gopher,
            "from paper.scripts.benchmark import rules",
            (),
        ),
        Spec(
            "c4_rules",
            "baseline",
            _c4(files / "ldnoobw_en.txt"),
            "from paper.scripts.benchmark import rules",
            (),
        ),
    ]
    return {spec.name: spec for spec in out}
