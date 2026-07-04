from pygarble import GarbleDetector, Strategy


class TestStrategies:
    def test_pattern_matching_detector(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING, threshold=0.2)
        assert detector.predict("AAAAA") is True
        assert detector.predict("asdfghjkl") is True
        assert detector.predict("normal text") is False

    def test_entropy_based_detector(self):
        detector = GarbleDetector(
            Strategy.ENTROPY_BASED, entropy_threshold=2.0
        )
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("normal text") is False

    def test_pattern_matching_proba(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        proba_pattern = detector.predict_proba("AAAAA")
        proba_normal = detector.predict_proba("normal text")
        assert proba_pattern > proba_normal
        assert 0.0 <= proba_pattern <= 1.0
        assert 0.0 <= proba_normal <= 1.0

    def test_entropy_based_proba(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        proba_repeated = detector.predict_proba("aaaaaaa")
        proba_diverse = detector.predict_proba("normal text")
        assert proba_repeated > proba_diverse
        assert 0.0 <= proba_repeated <= 1.0
        assert 0.0 <= proba_diverse <= 1.0


class TestStrategy:
    def test_strategy_enum_values(self):
        assert Strategy.PATTERN_MATCHING.value == "pattern_matching"
        assert Strategy.ENTROPY_BASED.value == "entropy_based"
        assert Strategy.VOWEL_RATIO.value == "vowel_ratio"

    def test_strategy_enum_size(self):
        assert len(Strategy) == 26
