from pygarble import Strategy


class TestCoreFunctionality:
    def test_basic_import(self):
        from pygarble import GarbleDetector, Strategy

        assert GarbleDetector is not None
        assert Strategy is not None

    def test_strategy_enum_values(self):
        assert Strategy.PATTERN_MATCHING.value == "pattern_matching"
        assert Strategy.ENTROPY_BASED.value == "entropy_based"
        assert Strategy.VOWEL_RATIO.value == "vowel_ratio"

    def test_strategy_enum_size(self):
        assert len(Strategy) == 26

    def test_all_strategies_importable(self):
        from pygarble.strategies import (
            EntropyBasedStrategy,
            MarkovChainStrategy,
            PatternMatchingStrategy,
            WordLookupStrategy,
        )

        assert PatternMatchingStrategy is not None
        assert EntropyBasedStrategy is not None
        assert MarkovChainStrategy is not None
        assert WordLookupStrategy is not None
