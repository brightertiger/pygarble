from pygarble import Strategy


class TestCoreFunctionality:
    def test_basic_import(self):
        from pygarble import GarbleDetector, Strategy

        assert GarbleDetector is not None
        assert Strategy is not None

    def test_strategy_enum_values(self):
        assert Strategy.CHARACTER_FREQUENCY.value == "character_frequency"
        assert Strategy.WORD_LENGTH.value == "word_length"
        assert Strategy.PATTERN_MATCHING.value == "pattern_matching"
        assert Strategy.STATISTICAL_ANALYSIS.value == "statistical_analysis"
        assert Strategy.ENTROPY_BASED.value == "entropy_based"
        assert Strategy.VOWEL_RATIO.value == "vowel_ratio"

    def test_all_strategies_importable(self):
        from pygarble.strategies import (
            CharacterFrequencyStrategy,
            EntropyBasedStrategy,
            PatternMatchingStrategy,
            StatisticalAnalysisStrategy,
            WordLengthStrategy,
        )

        assert CharacterFrequencyStrategy is not None
        assert WordLengthStrategy is not None
        assert PatternMatchingStrategy is not None
        assert StatisticalAnalysisStrategy is not None
        assert EntropyBasedStrategy is not None
