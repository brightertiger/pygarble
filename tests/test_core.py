class TestCoreFunctionality:
    def test_basic_import(self):
        from pygarble import GarbleDetector, Strategy

        assert GarbleDetector is not None
        assert Strategy is not None

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
