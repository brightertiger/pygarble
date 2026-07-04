import pytest

from pygarble import GarbleDetector, Strategy


class TestGarbleDetector:
    def test_init_with_strategy(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        assert detector.strategy == Strategy.WORD_LOOKUP
        assert detector.kwargs == {}

    def test_init_with_kwargs(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, min_word_length=5)
        assert detector.strategy == Strategy.WORD_LOOKUP
        assert detector.kwargs == {"min_word_length": 5}

    def test_init_with_threshold(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, threshold=0.3)
        assert detector.threshold == 0.3

    def test_init_with_threshold_and_kwargs(self):
        detector = GarbleDetector(
            Strategy.WORD_LOOKUP, threshold=0.7, min_word_length=5
        )
        assert detector.threshold == 0.7
        assert detector.kwargs == {"min_word_length": 5}

    def test_predict_single_string(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        result = detector.predict("aaaaaaa")
        assert isinstance(result, bool)

    def test_predict_list_of_strings(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        texts = ["aaaaaaa", "normal text", "bbbbbbb"]
        results = detector.predict(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, bool) for r in results)

    def test_predict_invalid_input(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        with pytest.raises(TypeError):
            detector.predict(123)

    def test_predict_proba_single_string(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        result = detector.predict_proba("aaaaaaa")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_proba_list_of_strings(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        texts = ["aaaaaaa", "normal text", "bbbbbbb"]
        results = detector.predict_proba(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, float) and 0.0 <= r <= 1.0 for r in results)

    def test_predict_proba_invalid_input(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        with pytest.raises(TypeError):
            detector.predict_proba(123)

    def test_predict_with_threshold(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, threshold=0.3)
        proba = detector.predict_proba("aaaaaaa")
        prediction = detector.predict("aaaaaaa")
        assert prediction == (proba >= 0.3)

    def test_not_implemented_strategy(self):
        from enum import Enum

        class UnsupportedStrategy(Enum):
            UNSUPPORTED = "unsupported"

        with pytest.raises(NotImplementedError):
            GarbleDetector(UnsupportedStrategy.UNSUPPORTED)
