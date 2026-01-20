import concurrent.futures
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .strategies import (
    BaseStrategy,
    CharacterFrequencyStrategy,
    EntropyBasedStrategy,
    PatternMatchingStrategy,
    StatisticalAnalysisStrategy,
    WordLengthStrategy,
    EnglishWordValidationStrategy,
    VowelRatioStrategy,
    KeyboardPatternStrategy,
    MarkovChainStrategy,
    NGramFrequencyStrategy,
    WordLookupStrategy,
    SymbolRatioStrategy,
    RepetitionStrategy,
    HexStringStrategy,
    CompressionRatioStrategy,
    MojibakeStrategy,
    PronouncabilityStrategy,
    UnicodeScriptStrategy,
    BigramProbabilityStrategy,
    LetterPositionStrategy,
    ConsonantSequenceStrategy,
    VowelPatternStrategy,
    LetterFrequencyStrategy,
    RareTrigramStrategy,
)


class Strategy(Enum):
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    ENGLISH_WORD_VALIDATION = "english_word_validation"
    VOWEL_RATIO = "vowel_ratio"
    KEYBOARD_PATTERN = "keyboard_pattern"
    MARKOV_CHAIN = "markov_chain"
    NGRAM_FREQUENCY = "ngram_frequency"
    WORD_LOOKUP = "word_lookup"
    SYMBOL_RATIO = "symbol_ratio"
    REPETITION = "repetition"
    HEX_STRING = "hex_string"
    COMPRESSION_RATIO = "compression_ratio"
    MOJIBAKE = "mojibake"
    PRONOUNCEABILITY = "pronounceability"
    UNICODE_SCRIPT = "unicode_script"
    BIGRAM_PROBABILITY = "bigram_probability"
    LETTER_POSITION = "letter_position"
    CONSONANT_SEQUENCE = "consonant_sequence"
    VOWEL_PATTERN = "vowel_pattern"
    LETTER_FREQUENCY = "letter_frequency"
    RARE_TRIGRAM = "rare_trigram"


STRATEGY_MAP: Dict[Strategy, Type[BaseStrategy]] = {
    Strategy.CHARACTER_FREQUENCY: CharacterFrequencyStrategy,
    Strategy.WORD_LENGTH: WordLengthStrategy,
    Strategy.PATTERN_MATCHING: PatternMatchingStrategy,
    Strategy.STATISTICAL_ANALYSIS: StatisticalAnalysisStrategy,
    Strategy.ENTROPY_BASED: EntropyBasedStrategy,
    Strategy.ENGLISH_WORD_VALIDATION: EnglishWordValidationStrategy,
    Strategy.VOWEL_RATIO: VowelRatioStrategy,
    Strategy.KEYBOARD_PATTERN: KeyboardPatternStrategy,
    Strategy.MARKOV_CHAIN: MarkovChainStrategy,
    Strategy.NGRAM_FREQUENCY: NGramFrequencyStrategy,
    Strategy.WORD_LOOKUP: WordLookupStrategy,
    Strategy.SYMBOL_RATIO: SymbolRatioStrategy,
    Strategy.REPETITION: RepetitionStrategy,
    Strategy.HEX_STRING: HexStringStrategy,
    Strategy.COMPRESSION_RATIO: CompressionRatioStrategy,
    Strategy.MOJIBAKE: MojibakeStrategy,
    Strategy.PRONOUNCEABILITY: PronouncabilityStrategy,
    Strategy.UNICODE_SCRIPT: UnicodeScriptStrategy,
    Strategy.BIGRAM_PROBABILITY: BigramProbabilityStrategy,
    Strategy.LETTER_POSITION: LetterPositionStrategy,
    Strategy.CONSONANT_SEQUENCE: ConsonantSequenceStrategy,
    Strategy.VOWEL_PATTERN: VowelPatternStrategy,
    Strategy.LETTER_FREQUENCY: LetterFrequencyStrategy,
    Strategy.RARE_TRIGRAM: RareTrigramStrategy,
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
        timeout_per_text = self.kwargs.get("timeout_per_text", 30.0)
        total_timeout = timeout_per_text * len(texts)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [executor.submit(process_func, text) for text in texts]
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=total_timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    # Return default value on timeout
                    results.append(False if process_func == self._process_text_predict else 0.0)
                except Exception:
                    # Return default value on any exception
                    results.append(False if process_func == self._process_text_predict else 0.0)
            return results

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
        if voting not in ("majority", "average", "weighted", "any", "all"):
            raise ValueError(
                "voting must be 'majority', 'average', 'weighted', 'any', or 'all'"
            )
        if voting == "weighted" and weights is None:
            raise ValueError("weights required when voting='weighted'")

        if strategies is None:
            # High-precision default: uses strategies with 90%+ precision
            # Combined with majority voting for reliable detection
            strategies = [
                Strategy.MARKOV_CHAIN,       # 95% precision, 61% recall
                Strategy.WORD_LOOKUP,        # 89% precision, 51% recall
                Strategy.NGRAM_FREQUENCY,    # 88% precision, 47% recall
                Strategy.BIGRAM_PROBABILITY, # 100% precision, 25% recall
                Strategy.LETTER_POSITION,    # 93% precision, 35% recall
            ]

        if not strategies:
            raise ValueError("strategies must contain at least one strategy")

        if weights is not None:
            if len(weights) != len(strategies):
                raise ValueError("weights must have same length as strategies")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            if sum(weights) == 0:
                raise ValueError("weights must not all be zero")

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
        elif self.voting == "any":
            # High recall: if ANY strategy flags as garbled, return True
            return any(d.predict(text) for d in self._detectors)
        elif self.voting == "all":
            # High precision: ALL strategies must agree
            return all(d.predict(text) for d in self._detectors)
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
        if not self._detectors:
            return 0.0  # Defensive check (should never happen due to validation)

        probas = [d.predict_proba(text) for d in self._detectors]

        if self.voting == "weighted":
            total_weight = sum(self.weights)
            return sum(p * w for p, w in zip(probas, self.weights)) / total_weight
        elif self.voting == "any":
            # Return max probability (most suspicious signal)
            return max(probas)
        elif self.voting == "all":
            # Return min probability (least suspicious signal)
            return min(probas)
        else:
            return sum(probas) / len(probas)
