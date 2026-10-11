"""Public strategy names and lazy factories; no model data loads here."""

from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Type, cast

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
    CROSS_PARSING = "cross_parsing"
    PRIMED_COMPRESSION = "primed_compression"
    NGRAM_RANK = "ngram_rank"
    PERMUTATION_TEST = "permutation_test"


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
    Strategy.CROSS_PARSING: ("cross_parsing", "CrossParsingStrategy"),
    Strategy.PRIMED_COMPRESSION: (
        "primed_compression",
        "PrimedCompressionStrategy",
    ),
    Strategy.NGRAM_RANK: ("ngram_rank", "NGramRankStrategy"),
    Strategy.PERMUTATION_TEST: ("permutation_test", "PermutationTestStrategy"),
}


if TYPE_CHECKING:
    STRATEGY_MAP: Dict[Strategy, Type[BaseStrategy]]


def strategy_class(strategy: Strategy) -> Type[BaseStrategy]:
    """Import one strategy class; the rest stay unloaded."""
    built = globals().get("STRATEGY_MAP")
    if built is not None:
        # Once built, the public dict is authoritative, as in 0.8.0.
        return cast(Type[BaseStrategy], built[strategy])
    module, name = _IMPLEMENTATIONS[strategy]
    return cast(
        Type[BaseStrategy],
        getattr(import_module(".strategies." + module, __package__), name),
    )


def __getattr__(name: str) -> Any:
    # STRATEGY_MAP is a plain dict, built on first access because building
    # it imports every strategy module.
    if name != "STRATEGY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    built = {key: strategy_class(key) for key in _IMPLEMENTATIONS}
    return globals().setdefault("STRATEGY_MAP", built)


def __dir__() -> List[str]:
    return sorted(set(globals()) | {"STRATEGY_MAP"})
