"""Accepted legacy strategy settings. Unknown settings warn before removal."""

import os
import sys
import warnings
from types import FrameType
from typing import Any, FrozenSet, List, Mapping, Optional

PARAMETERS = {
    "PatternMatchingStrategy": [
        "max_string_length",
        "override_defaults",
        "patterns",
    ],
    "WordLookupStrategy": [
        "max_string_length",
        "min_word_length",
        "unknown_threshold",
    ],
    "HexStringStrategy": [
        "hex_ratio_threshold",
        "max_string_length",
        "min_hex_length",
    ],
    "UnicodeScriptStrategy": [
        "check_homoglyphs",
        "homoglyph_threshold",
        "max_scripts",
        "max_string_length",
    ],
    "MarkovChainStrategy": [
        "max_string_length",
        "min_length",
        "threshold_per_char",
    ],
    "VowelRatioStrategy": [
        "consonant_cluster_len",
        "max_string_length",
        "max_vowel_ratio",
        "min_length",
        "min_vowel_ratio",
    ],
    "VowelPatternStrategy": [
        "max_string_length",
        "max_vowel_run",
        "min_length",
    ],
    "FunctionWordDensityStrategy": [
        "max_string_length",
        "min_ratio",
        "min_word_length",
        "min_words",
    ],
    "ZipfConformityStrategy": [
        "hapax_threshold",
        "max_string_length",
        "min_words",
        "ttr_threshold",
    ],
    "BigramProbabilityStrategy": [
        "max_string_length",
        "min_length",
        "threshold",
    ],
    "WordAnomalyStrategy": [
        "anomaly_weight",
        "max_string_length",
        "min_word_length",
        "word_log_prob_threshold",
    ],
    "ConsonantSequenceStrategy": [
        "max_consonants",
        "max_string_length",
        "min_length",
    ],
    "LocalAnomalyStrategy": [
        "max_string_length",
        "min_word_length",
        "window_words",
        "word_log_prob_threshold",
    ],
    "LetterPositionStrategy": [
        "max_string_length",
        "min_word_length",
        "threshold",
    ],
    "LetterFrequencyStrategy": [
        "deviation_threshold",
        "max_string_length",
        "min_length",
    ],
    "ControlCharactersStrategy": ["max_combining_run", "max_string_length"],
    "AffixDetectionStrategy": [
        "max_string_length",
        "min_affix_ratio",
        "min_analyzable_words",
        "min_stem_length",
        "min_word_length",
    ],
    "KeyboardPatternStrategy": ["max_string_length"],
    "RepetitionStrategy": [
        "diversity_threshold",
        "max_char_repeat",
        "max_pattern_repeat",
        "max_string_length",
    ],
    "PronouncabilityStrategy": [
        "forbidden_cluster_threshold",
        "max_string_length",
        "min_word_length",
        "vowel_min_ratio",
    ],
    "NGramFrequencyStrategy": [
        "common_ratio_threshold",
        "max_string_length",
        "min_length",
    ],
    "KeyboardAdjacencyStrategy": [
        "chain_threshold",
        "keyboard_layout",
        "max_string_length",
        "min_word_length",
        "row_run_threshold",
    ],
    "EntropyBasedStrategy": ["max_string_length"],
    "MojibakeStrategy": [
        "check_replacement_char",
        "max_string_length",
        "pattern_threshold",
        "ratio_threshold",
    ],
    "RareTrigramStrategy": ["max_string_length", "min_length", "threshold"],
    "SymbolRatioStrategy": [
        "allow_spaces",
        "count_digits",
        "max_string_length",
        "min_length",
        "symbol_threshold",
    ],
    "LogLikelihoodRatioStrategy": [
        "llr_midpoint",
        "llr_scale",
        "max_string_length",
        "min_bigrams",
    ],
    "WordCollocationStrategy": [
        "max_string_length",
        "min_words",
        "zero_collocation_min_words",
    ],
    "CrossParsingStrategy": [
        "max_string_length",
        "midpoint",
        "min_length",
        "scale",
    ],
    "PrimedCompressionStrategy": [
        "max_string_length",
        "midpoint",
        "min_length",
        "scale",
    ],
    "NGramRankStrategy": [
        "max_string_length",
        "midpoint",
        "min_length",
        "scale",
    ],
    "PermutationTestStrategy": [
        "max_string_length",
        "midpoint",
        "min_length",
        "scale",
        "shuffles",
    ],
}


def accepted_options(strategy: str) -> Optional[FrozenSet[str]]:
    """Settings a strategy class accepts, or None if unrestricted."""
    accepted = PARAMETERS.get(strategy)
    return None if accepted is None else frozenset(accepted)


def unknown_options(strategy: str, options: Mapping[str, Any]) -> List[str]:
    accepted = accepted_options(strategy)
    if accepted is None:
        return []
    return sorted(set(options) - accepted)


def _external_stacklevel() -> int:
    """Stack level of the first frame outside the pygarble package."""
    # sys._getframe is CPython/PyPy-specific; both are supported targets.
    package = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    )
    frame: Optional[FrameType] = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(
        frame.f_code.co_filename
    ).startswith(package):
        frame = frame.f_back
        level += 1
    return level


def warn_unknown_options(strategy: str, unknown: List[str]) -> None:
    if not unknown:
        return
    warnings.warn(
        f"Unknown settings for {strategy}: {', '.join(unknown)}. "
        "Unknown settings will become errors in a future release.",
        FutureWarning,
        stacklevel=_external_stacklevel(),
    )


def validate_options(strategy: str, options: Mapping[str, Any]) -> None:
    warn_unknown_options(strategy, unknown_options(strategy, options))
