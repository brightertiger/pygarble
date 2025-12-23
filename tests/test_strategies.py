from pygarble import GarbleDetector, Strategy


class TestStrategies:
    def test_character_frequency_detector(self):
        detector = GarbleDetector(
            Strategy.CHARACTER_FREQUENCY, frequency_threshold=0.3
        )
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("normal text") is False

    def test_word_length_detector(self):
        detector = GarbleDetector(
            Strategy.WORD_LENGTH, max_word_length=5, threshold=0.5
        )
        assert detector.predict("supercalifragilisticexpialidocious") is True
        assert detector.predict("short words") is False

    def test_pattern_matching_detector(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING, threshold=0.2)
        assert detector.predict("AAAAA") is True
        assert detector.predict("asdfghjkl") is True
        assert detector.predict("normal text") is False

    def test_statistical_analysis_detector(self):
        detector = GarbleDetector(
            Strategy.STATISTICAL_ANALYSIS, alpha_threshold=0.3
        )
        assert detector.predict("123456789") is True
        assert detector.predict("normal text") is False

    def test_entropy_based_detector(self):
        detector = GarbleDetector(
            Strategy.ENTROPY_BASED, entropy_threshold=2.0
        )
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("normal text") is False

    def test_character_frequency_proba(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        proba_high = detector.predict_proba("aaaaaaa")
        proba_low = detector.predict_proba("normal text")
        assert proba_high > proba_low
        assert 0.0 <= proba_high <= 1.0
        assert 0.0 <= proba_low <= 1.0

    def test_word_length_proba(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH, max_word_length=10)
        proba_long = detector.predict_proba(
            "supercalifragilisticexpialidocious"
        )
        proba_short = detector.predict_proba("short words")
        assert proba_long > proba_short
        assert 0.0 <= proba_long <= 1.0
        assert 0.0 <= proba_short <= 1.0

    def test_pattern_matching_proba(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        proba_pattern = detector.predict_proba("AAAAA")
        proba_normal = detector.predict_proba("normal text")
        assert proba_pattern > proba_normal
        assert 0.0 <= proba_pattern <= 1.0
        assert 0.0 <= proba_normal <= 1.0

    def test_statistical_analysis_proba(self):
        detector = GarbleDetector(Strategy.STATISTICAL_ANALYSIS)
        proba_numbers = detector.predict_proba("123456789")
        proba_text = detector.predict_proba("normal text")
        assert proba_numbers > proba_text
        assert 0.0 <= proba_numbers <= 1.0
        assert 0.0 <= proba_text <= 1.0

    def test_entropy_based_proba(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        proba_repeated = detector.predict_proba("aaaaaaa")
        proba_diverse = detector.predict_proba("normal text")
        assert proba_repeated > proba_diverse
        assert 0.0 <= proba_repeated <= 1.0
        assert 0.0 <= proba_diverse <= 1.0

    def test_english_word_validation_detector(self):
        detector = GarbleDetector(
            Strategy.ENGLISH_WORD_VALIDATION, threshold=0.7
        )
        assert detector.predict("hello world this is normal text") is False
        assert detector.predict("asdfghjkl mnbvcxz lkjhgfds") is True

    def test_english_word_validation_proba(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        proba_valid = detector.predict_proba("hello world this is normal text")
        proba_invalid = detector.predict_proba("asdfghjkl mnbvcxz lkjhgfds")
        assert proba_invalid > proba_valid
        assert 0.0 <= proba_valid <= 1.0
        assert 0.0 <= proba_invalid <= 1.0

    def test_english_word_validation_mixed_content(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.3)
        assert detector.predict("hello world asdfghjkl") is True
        assert detector.predict("asdfghjkl mnbvcxz hello") is True

    def test_english_word_validation_threshold_adjustment(self):
        detector_strict = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.9)
        detector_loose = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.3)
        
        # Text with 50% invalid words
        mixed_text = "hello world asdfghjkl mnbvcxz"
        
        # With strict threshold (90%), 50% invalid words should NOT be detected as garbled
        assert detector_strict.predict(mixed_text) is False
        # With loose threshold (30%), 50% invalid words should be detected as garbled
        assert detector_loose.predict(mixed_text) is True
        
        # Test with mostly valid text
        mostly_valid_text = "hello world this is normal text asdfghjkl"
        assert detector_strict.predict(mostly_valid_text) is False
        assert detector_loose.predict(mostly_valid_text) is False

    def test_english_word_validation_empty_and_whitespace(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        assert detector.predict("") is False
        assert detector.predict("   ") is False
        assert detector.predict_proba("") == 0.0
        assert detector.predict_proba("   ") == 0.0

    def test_english_word_validation_numbers_and_symbols(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        assert detector.predict("123 456 789") is False
        assert detector.predict("hello 123 world") is False
        assert detector.predict("!@#$%^&*()") is False

    def test_english_word_validation_case_insensitive(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        assert detector.predict("HELLO WORLD") is False
        assert detector.predict("Hello World") is False
        assert detector.predict("hello world") is False

    def test_english_word_validation_single_word(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        assert detector.predict("hello") is False
        assert detector.predict("asdfghjkl") is True
        assert detector.predict_proba("hello") == 0.0
        assert detector.predict_proba("asdfghjkl") == 1.0


class TestStrategy:
    def test_strategy_enum_values(self):
        assert Strategy.CHARACTER_FREQUENCY.value == "character_frequency"
        assert Strategy.WORD_LENGTH.value == "word_length"
        assert Strategy.PATTERN_MATCHING.value == "pattern_matching"
        assert Strategy.STATISTICAL_ANALYSIS.value == "statistical_analysis"
        assert Strategy.ENTROPY_BASED.value == "entropy_based"
        assert Strategy.LANGUAGE_DETECTION.value == "language_detection"
        assert Strategy.ENGLISH_WORD_VALIDATION.value == "english_word_validation"
        assert Strategy.VOWEL_RATIO.value == "vowel_ratio"
