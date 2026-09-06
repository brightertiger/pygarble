"""Accepted legacy strategy settings. Unknown settings warn before removal."""

import warnings
from typing import Any, Mapping

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
}


def validate_options(strategy: str, options: Mapping[str, Any]) -> None:
    accepted = PARAMETERS.get(strategy)
    if accepted is None:
        return
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        warnings.warn(
            f"Unknown settings for {strategy}: {', '.join(unknown)}; "
            "use strategy_kwargs to configure ensemble members separately. "
            "Unknown settings will become errors in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
