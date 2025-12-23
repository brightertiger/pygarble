import pytest
from pygarble import GarbleDetector, Strategy, EnsembleDetector


class TestVowelRatioStrategy:
    def test_vowel_ratio_normal_text(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("hello world") is False
        assert detector.predict("this is normal english text") is False

    def test_vowel_ratio_no_vowels(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("bcdfghjklmnpqrstvwxyz") is True
        assert detector.predict_proba("bcdfghjklmnpqrstvwxyz") > 0.5

    def test_vowel_ratio_all_vowels(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("aeiouaeiou") is True
        assert detector.predict_proba("aeiouaeiou") > 0.5

    def test_vowel_ratio_proba_range(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        proba = detector.predict_proba("hello world")
        assert 0.0 <= proba <= 1.0

    def test_vowel_ratio_custom_thresholds(self):
        detector = GarbleDetector(
            Strategy.VOWEL_RATIO,
            min_vowel_ratio=0.3,
            max_vowel_ratio=0.5
        )
        assert detector.predict("aeiouaeiou") is True

    def test_vowel_ratio_empty_string(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_vowel_ratio_numbers_only(self):
        detector = GarbleDetector(Strategy.VOWEL_RATIO)
        assert detector.predict("123456789") is False
        assert detector.predict_proba("123456789") == 0.0


class TestEnsembleDetector:
    def test_ensemble_default_strategies(self):
        detector = EnsembleDetector()
        assert detector.predict("hello world") is False
        assert detector.predict("AAAAAAAAAA @@@") is True

    def test_ensemble_custom_strategies(self):
        detector = EnsembleDetector(
            strategies=[
                Strategy.CHARACTER_FREQUENCY,
                Strategy.ENTROPY_BASED,
            ]
        )
        assert detector.predict("normal text") is False

    def test_ensemble_majority_voting(self):
        detector = EnsembleDetector(
            strategies=[
                Strategy.CHARACTER_FREQUENCY,
                Strategy.ENTROPY_BASED,
                Strategy.PATTERN_MATCHING,
            ],
            voting="majority"
        )
        result = detector.predict("AAAAA")
        assert isinstance(result, bool)

    def test_ensemble_average_voting(self):
        detector = EnsembleDetector(
            strategies=[Strategy.CHARACTER_FREQUENCY, Strategy.ENTROPY_BASED],
            voting="average"
        )
        proba = detector.predict_proba("hello world")
        assert 0.0 <= proba <= 1.0

    def test_ensemble_weighted_voting(self):
        detector = EnsembleDetector(
            strategies=[Strategy.CHARACTER_FREQUENCY, Strategy.ENTROPY_BASED],
            voting="weighted",
            weights=[0.7, 0.3]
        )
        proba = detector.predict_proba("hello world")
        assert 0.0 <= proba <= 1.0

    def test_ensemble_batch_processing(self):
        detector = EnsembleDetector()
        texts = ["hello world", "aaaaaaa", "normal text"]
        results = detector.predict(texts)
        assert len(results) == 3
        assert isinstance(results[0], bool)

    def test_ensemble_proba_batch(self):
        detector = EnsembleDetector()
        texts = ["hello world", "aaaaaaa"]
        probas = detector.predict_proba(texts)
        assert len(probas) == 2
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_ensemble_invalid_voting(self):
        with pytest.raises(ValueError, match="voting must be"):
            EnsembleDetector(voting="invalid")

    def test_ensemble_weighted_without_weights(self):
        with pytest.raises(ValueError, match="weights required"):
            EnsembleDetector(voting="weighted")

    def test_ensemble_weights_length_mismatch(self):
        with pytest.raises(ValueError, match="weights must have same length"):
            EnsembleDetector(
                strategies=[Strategy.CHARACTER_FREQUENCY, Strategy.ENTROPY_BASED],
                voting="weighted",
                weights=[0.5]
            )


class TestValidation:
    def test_threshold_validation_low(self):
        with pytest.raises(ValueError, match="threshold must be"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=-0.1)

    def test_threshold_validation_high(self):
        with pytest.raises(ValueError, match="threshold must be"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=1.5)

    def test_threshold_validation_edge_low(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.0)
        assert detector.threshold == 0.0

    def test_threshold_validation_edge_high(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=1.0)
        assert detector.threshold == 1.0

    def test_threads_validation(self):
        with pytest.raises(ValueError, match="threads must be"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threads=0)

    def test_threads_validation_negative(self):
        with pytest.raises(ValueError, match="threads must be"):
            GarbleDetector(Strategy.CHARACTER_FREQUENCY, threads=-1)


class TestInputTypeValidation:
    def test_predict_invalid_type_int(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="must be a string"):
            detector.predict(123)

    def test_predict_invalid_type_dict(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="must be a string"):
            detector.predict({"text": "hello"})

    def test_predict_proba_invalid_type(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        with pytest.raises(TypeError, match="must be a string"):
            detector.predict_proba(123)

    def test_ensemble_predict_invalid_type(self):
        detector = EnsembleDetector()
        with pytest.raises(TypeError, match="must be a string"):
            detector.predict(123)


class TestPatternMatchingCompilation:
    def test_pattern_compilation_happens_once(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        compiled = detector._strategy_instance._compiled_patterns
        assert all(hasattr(p, "search") for p in compiled.values())

    def test_custom_patterns_compiled(self):
        detector = GarbleDetector(
            Strategy.PATTERN_MATCHING,
            patterns={"custom": r"\d{3}-\d{4}"}
        )
        compiled = detector._strategy_instance._compiled_patterns
        assert "custom" in compiled
        assert compiled["custom"].search("123-4567") is not None


class TestBatchProcessing:
    def test_batch_with_threads(self):
        detector = GarbleDetector(
            Strategy.CHARACTER_FREQUENCY,
            threads=2
        )
        texts = ["hello world"] * 20
        results = detector.predict(texts)
        assert len(results) == 20

    def test_batch_proba_with_threads(self):
        detector = GarbleDetector(
            Strategy.ENTROPY_BASED,
            threads=2
        )
        texts = ["hello world", "aaaaaaa"] * 10
        probas = detector.predict_proba(texts)
        assert len(probas) == 20
        assert all(0.0 <= p <= 1.0 for p in probas)

