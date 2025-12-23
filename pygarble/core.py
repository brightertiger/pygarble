import concurrent.futures
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .strategies import (
    BaseStrategy,
    CharacterFrequencyStrategy,
    EntropyBasedStrategy,
    LanguageDetectionStrategy,
    PatternMatchingStrategy,
    StatisticalAnalysisStrategy,
    WordLengthStrategy,
    EnglishWordValidationStrategy,
    VowelRatioStrategy,
    KeyboardPatternStrategy,
)


class Strategy(Enum):
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    LANGUAGE_DETECTION = "language_detection"
    ENGLISH_WORD_VALIDATION = "english_word_validation"
    VOWEL_RATIO = "vowel_ratio"
    KEYBOARD_PATTERN = "keyboard_pattern"


STRATEGY_MAP: Dict[Strategy, Type[BaseStrategy]] = {
    Strategy.CHARACTER_FREQUENCY: CharacterFrequencyStrategy,
    Strategy.WORD_LENGTH: WordLengthStrategy,
    Strategy.PATTERN_MATCHING: PatternMatchingStrategy,
    Strategy.STATISTICAL_ANALYSIS: StatisticalAnalysisStrategy,
    Strategy.ENTROPY_BASED: EntropyBasedStrategy,
    Strategy.LANGUAGE_DETECTION: LanguageDetectionStrategy,
    Strategy.ENGLISH_WORD_VALIDATION: EnglishWordValidationStrategy,
    Strategy.VOWEL_RATIO: VowelRatioStrategy,
    Strategy.KEYBOARD_PATTERN: KeyboardPatternStrategy,
}


class GarbleDetector:
    def __init__(
        self,
        strategy: Strategy,
        threshold: float = 0.5,
        threads: Optional[int] = None,
        **kwargs: Any,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if threads is not None and threads < 1:
            raise ValueError("threads must be a positive integer")

        self.strategy = strategy
        self.threshold = threshold
        self.threads = threads
        self.kwargs = kwargs
        self._strategy_instance = self._create_strategy_instance()

    def _create_strategy_instance(self) -> BaseStrategy:
        if self.strategy not in STRATEGY_MAP:
            raise NotImplementedError(
                f"Strategy {self.strategy.value} is not implemented"
            )

        strategy_class = STRATEGY_MAP[self.strategy]
        return strategy_class(**self.kwargs)

    def _process_text_proba(self, text: str) -> float:
        return self._strategy_instance.predict_proba(text)

    def _process_text_predict(self, text: str) -> bool:
        proba = self._strategy_instance.predict_proba(text)
        return proba >= self.threshold

    def _process_batch_threaded(
        self, texts: List[str], process_func: Callable[[str], Any]
    ) -> List[Any]:
        if self.threads is None or self.threads <= 1 or len(texts) < 10:
            return [process_func(text) for text in texts]

        max_workers = min(self.threads, len(texts))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [executor.submit(process_func, text) for text in texts]
            return [future.result() for future in futures]

    def predict(self, X: Union[str, List[str]]) -> Union[bool, List[bool]]:
        if isinstance(X, str):
            proba = self._strategy_instance.predict_proba(X)
            return proba >= self.threshold
        elif isinstance(X, list):
            return self._process_batch_threaded(X, self._process_text_predict)
        else:
            raise TypeError("Input must be a string or list of strings")

    def predict_proba(
        self, X: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        if isinstance(X, str):
            return self._strategy_instance.predict_proba(X)
        elif isinstance(X, list):
            return self._process_batch_threaded(X, self._process_text_proba)
        else:
            raise TypeError("Input must be a string or list of strings")


class EnsembleDetector:
    def __init__(
        self,
        strategies: Optional[List[Strategy]] = None,
        threshold: float = 0.5,
        voting: str = "majority",
        weights: Optional[List[float]] = None,
        threads: Optional[int] = None,
        **kwargs: Any,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if voting not in ("majority", "average", "weighted"):
            raise ValueError("voting must be 'majority', 'average', or 'weighted'")
        if voting == "weighted" and weights is None:
            raise ValueError("weights required when voting='weighted'")

        if strategies is None:
            strategies = [
                Strategy.KEYBOARD_PATTERN,
                Strategy.ENTROPY_BASED,
                Strategy.PATTERN_MATCHING,
                Strategy.VOWEL_RATIO,
                Strategy.ENGLISH_WORD_VALIDATION,
            ]

        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights must have same length as strategies")

        self.strategies = strategies
        self.threshold = threshold
        self.voting = voting
        self.weights = weights or [1.0] * len(strategies)
        self.threads = threads
        self.kwargs = kwargs

        self._detectors = [
            GarbleDetector(s, threshold=threshold, threads=threads, **kwargs)
            for s in strategies
        ]

    def predict(self, X: Union[str, List[str]]) -> Union[bool, List[bool]]:
        if isinstance(X, str):
            return self._predict_single(X)
        elif isinstance(X, list):
            return [self._predict_single(text) for text in X]
        else:
            raise TypeError("Input must be a string or list of strings")

    def _predict_single(self, text: str) -> bool:
        if self.voting == "majority":
            votes = sum(d.predict(text) for d in self._detectors)
            return votes > len(self._detectors) / 2
        else:
            proba = self._predict_proba_single(text)
            return proba >= self.threshold

    def predict_proba(
        self, X: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        if isinstance(X, str):
            return self._predict_proba_single(X)
        elif isinstance(X, list):
            return [self._predict_proba_single(text) for text in X]
        else:
            raise TypeError("Input must be a string or list of strings")

    def _predict_proba_single(self, text: str) -> float:
        probas = [d.predict_proba(text) for d in self._detectors]

        if self.voting == "weighted":
            total_weight = sum(self.weights)
            return sum(p * w for p, w in zip(probas, self.weights)) / total_weight
        else:
            return sum(probas) / len(probas)
