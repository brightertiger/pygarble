"""Public strategy names and lazy factories; no model data loads here."""

from collections.abc import Mapping
from enum import Enum
from importlib import import_module
from typing import Iterator, Type, cast

from .strategies.base import BaseStrategy


class Strategy(Enum):
    CONTROL_CHARACTERS = "control_characters"
    LOCAL_ANOMALY = "local_anomaly"
    PATTERN_MATCHING = "pattern_matching"
    ENTROPY_BASED = "entropy_based"
    VOWEL_RATIO = "vowel_ratio"
    KEYBOARD_PATTERN = "keyboard_pattern"
    MARKOV_CHAIN = "markov_chain"
    NGRAM_FREQUENCY = "ngram_frequency"
    WORD_LOOKUP = "word_lookup"
    SYMBOL_RATIO = "symbol_ratio"
    REPETITION = "repetition"
    HEX_STRING = "hex_string"
    MOJIBAKE = "mojibake"
    PRONOUNCEABILITY = "pronounceability"
    UNICODE_SCRIPT = "unicode_script"
    BIGRAM_PROBABILITY = "bigram_probability"
    LETTER_POSITION = "letter_position"
    CONSONANT_SEQUENCE = "consonant_sequence"
    VOWEL_PATTERN = "vowel_pattern"
    LETTER_FREQUENCY = "letter_frequency"
    RARE_TRIGRAM = "rare_trigram"
    FUNCTION_WORD_DENSITY = "function_word_density"
    AFFIX_DETECTION = "affix_detection"
    ZIPF_CONFORMITY = "zipf_conformity"
    WORD_COLLOCATION = "word_collocation"
    LOG_LIKELIHOOD_RATIO = "log_likelihood_ratio"
    WORD_ANOMALY = "word_anomaly"
    KEYBOARD_ADJACENCY = "keyboard_adjacency"


_IMPLEMENTATIONS = {
    Strategy.CONTROL_CHARACTERS: (
        "control_characters",
        "ControlCharactersStrategy",
    ),
    Strategy.LOCAL_ANOMALY: ("local_anomaly", "LocalAnomalyStrategy"),
    Strategy.PATTERN_MATCHING: ("pattern_matching", "PatternMatchingStrategy"),
    Strategy.ENTROPY_BASED: ("entropy_based", "EntropyBasedStrategy"),
    Strategy.VOWEL_RATIO: ("vowel_ratio", "VowelRatioStrategy"),
    Strategy.KEYBOARD_PATTERN: ("keyboard_pattern", "KeyboardPatternStrategy"),
    Strategy.MARKOV_CHAIN: ("markov_chain", "MarkovChainStrategy"),
    Strategy.NGRAM_FREQUENCY: ("ngram_frequency", "NGramFrequencyStrategy"),
    Strategy.WORD_LOOKUP: ("word_lookup", "WordLookupStrategy"),
    Strategy.SYMBOL_RATIO: ("symbol_ratio", "SymbolRatioStrategy"),
    Strategy.REPETITION: ("repetition", "RepetitionStrategy"),
    Strategy.HEX_STRING: ("hex_string", "HexStringStrategy"),
    Strategy.MOJIBAKE: ("mojibake", "MojibakeStrategy"),
    Strategy.PRONOUNCEABILITY: ("pronounceability", "PronouncabilityStrategy"),
    Strategy.UNICODE_SCRIPT: ("unicode_script", "UnicodeScriptStrategy"),
    Strategy.BIGRAM_PROBABILITY: (
        "bigram_probability",
        "BigramProbabilityStrategy",
    ),
    Strategy.LETTER_POSITION: ("letter_position", "LetterPositionStrategy"),
    Strategy.CONSONANT_SEQUENCE: (
        "consonant_sequence",
        "ConsonantSequenceStrategy",
    ),
    Strategy.VOWEL_PATTERN: ("vowel_pattern", "VowelPatternStrategy"),
    Strategy.LETTER_FREQUENCY: ("letter_frequency", "LetterFrequencyStrategy"),
    Strategy.RARE_TRIGRAM: ("rare_trigram", "RareTrigramStrategy"),
    Strategy.FUNCTION_WORD_DENSITY: (
        "function_word_density",
        "FunctionWordDensityStrategy",
    ),
    Strategy.AFFIX_DETECTION: ("affix_detection", "AffixDetectionStrategy"),
    Strategy.ZIPF_CONFORMITY: ("zipf_conformity", "ZipfConformityStrategy"),
    Strategy.WORD_COLLOCATION: ("word_collocation", "WordCollocationStrategy"),
    Strategy.LOG_LIKELIHOOD_RATIO: (
        "log_likelihood_ratio",
        "LogLikelihoodRatioStrategy",
    ),
    Strategy.WORD_ANOMALY: ("word_anomaly", "WordAnomalyStrategy"),
    Strategy.KEYBOARD_ADJACENCY: (
        "keyboard_adjacency",
        "KeyboardAdjacencyStrategy",
    ),
}


class _StrategyMap(Mapping):
    def __getitem__(self, key: Strategy) -> Type[BaseStrategy]:
        module, name = _IMPLEMENTATIONS[key]
        return cast(
            Type[BaseStrategy],
            getattr(import_module(".strategies." + module, "pygarble"), name),
        )

    def __iter__(self) -> Iterator[Strategy]:
        return iter(_IMPLEMENTATIONS)

    def __len__(self) -> int:
        return len(_IMPLEMENTATIONS)


STRATEGY_MAP = _StrategyMap()
