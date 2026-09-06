"""Lazy public exports; resources load only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .affix_detection import (
        AffixDetectionStrategy as AffixDetectionStrategy,
    )
    from .base import BaseStrategy as BaseStrategy
    from .bigram_probability import (
        BigramProbabilityStrategy as BigramProbabilityStrategy,
    )
    from .consonant_sequence import (
        ConsonantSequenceStrategy as ConsonantSequenceStrategy,
    )
    from .control_characters import (
        ControlCharactersStrategy as ControlCharactersStrategy,
    )
    from .entropy_based import EntropyBasedStrategy as EntropyBasedStrategy
    from .function_word_density import (
        FunctionWordDensityStrategy as FunctionWordDensityStrategy,
    )
    from .hex_string import HexStringStrategy as HexStringStrategy
    from .keyboard_adjacency import (
        KeyboardAdjacencyStrategy as KeyboardAdjacencyStrategy,
    )
    from .keyboard_pattern import (
        KeyboardPatternStrategy as KeyboardPatternStrategy,
    )
    from .letter_frequency import (
        LetterFrequencyStrategy as LetterFrequencyStrategy,
    )
    from .letter_position import (
        LetterPositionStrategy as LetterPositionStrategy,
    )
    from .local_anomaly import LocalAnomalyStrategy as LocalAnomalyStrategy
    from .log_likelihood_ratio import (
        LogLikelihoodRatioStrategy as LogLikelihoodRatioStrategy,
    )
    from .markov_chain import MarkovChainStrategy as MarkovChainStrategy
    from .mojibake import MojibakeStrategy as MojibakeStrategy
    from .ngram_frequency import (
        NGramFrequencyStrategy as NGramFrequencyStrategy,
    )
    from .pattern_matching import (
        PatternMatchingStrategy as PatternMatchingStrategy,
    )
    from .pronounceability import (
        PronouncabilityStrategy as PronouncabilityStrategy,
    )
    from .rare_trigram import RareTrigramStrategy as RareTrigramStrategy
    from .repetition import RepetitionStrategy as RepetitionStrategy
    from .symbol_ratio import SymbolRatioStrategy as SymbolRatioStrategy
    from .unicode_script import UnicodeScriptStrategy as UnicodeScriptStrategy
    from .vowel_pattern import VowelPatternStrategy as VowelPatternStrategy
    from .vowel_ratio import VowelRatioStrategy as VowelRatioStrategy
    from .word_anomaly import WordAnomalyStrategy as WordAnomalyStrategy
    from .word_collocation import (
        WordCollocationStrategy as WordCollocationStrategy,
    )
    from .word_lookup import WordLookupStrategy as WordLookupStrategy
    from .zipf_conformity import (
        ZipfConformityStrategy as ZipfConformityStrategy,
    )

_EXPORTS = {
    "ControlCharactersStrategy": "control_characters",
    "LocalAnomalyStrategy": "local_anomaly",
    "BaseStrategy": "base",
    "EntropyBasedStrategy": "entropy_based",
    "PatternMatchingStrategy": "pattern_matching",
    "VowelRatioStrategy": "vowel_ratio",
    "KeyboardPatternStrategy": "keyboard_pattern",
    "MarkovChainStrategy": "markov_chain",
    "NGramFrequencyStrategy": "ngram_frequency",
    "WordLookupStrategy": "word_lookup",
    "SymbolRatioStrategy": "symbol_ratio",
    "RepetitionStrategy": "repetition",
    "HexStringStrategy": "hex_string",
    "MojibakeStrategy": "mojibake",
    "PronouncabilityStrategy": "pronounceability",
    "UnicodeScriptStrategy": "unicode_script",
    "BigramProbabilityStrategy": "bigram_probability",
    "LetterPositionStrategy": "letter_position",
    "ConsonantSequenceStrategy": "consonant_sequence",
    "VowelPatternStrategy": "vowel_pattern",
    "LetterFrequencyStrategy": "letter_frequency",
    "RareTrigramStrategy": "rare_trigram",
    "FunctionWordDensityStrategy": "function_word_density",
    "AffixDetectionStrategy": "affix_detection",
    "ZipfConformityStrategy": "zipf_conformity",
    "WordCollocationStrategy": "word_collocation",
    "LogLikelihoodRatioStrategy": "log_likelihood_ratio",
    "WordAnomalyStrategy": "word_anomaly",
    "KeyboardAdjacencyStrategy": "keyboard_adjacency",
}
__all__ = [
    "ControlCharactersStrategy",
    "LocalAnomalyStrategy",
    "BaseStrategy",
    "EntropyBasedStrategy",
    "PatternMatchingStrategy",
    "VowelRatioStrategy",
    "KeyboardPatternStrategy",
    "MarkovChainStrategy",
    "NGramFrequencyStrategy",
    "WordLookupStrategy",
    "SymbolRatioStrategy",
    "RepetitionStrategy",
    "HexStringStrategy",
    "MojibakeStrategy",
    "PronouncabilityStrategy",
    "UnicodeScriptStrategy",
    "BigramProbabilityStrategy",
    "LetterPositionStrategy",
    "ConsonantSequenceStrategy",
    "VowelPatternStrategy",
    "LetterFrequencyStrategy",
    "RareTrigramStrategy",
    "FunctionWordDensityStrategy",
    "AffixDetectionStrategy",
    "ZipfConformityStrategy",
    "WordCollocationStrategy",
    "LogLikelihoodRatioStrategy",
    "WordAnomalyStrategy",
    "KeyboardAdjacencyStrategy",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
