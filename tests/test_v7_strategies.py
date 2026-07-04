"""Tests for the v0.7.0 strategies: LOG_LIKELIHOOD_RATIO, WORD_ANOMALY,
KEYBOARD_ADJACENCY, and the ensemble abstention mechanism."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy
from pygarble.strategies.keyboard_adjacency import KeyboardAdjacencyStrategy
from pygarble.strategies.log_likelihood_ratio import (
    LogLikelihoodRatioStrategy,
)
from pygarble.strategies.word_anomaly import WordAnomalyStrategy


class TestLogLikelihoodRatioStrategy:
    def setup_method(self):
        self.detector = GarbleDetector(Strategy.LOG_LIKELIHOOD_RATIO)

    def test_clean_english(self):
        assert self.detector.predict("the quick brown fox") is False
        assert self.detector.predict("Hello there.") is False
        assert self.detector.predict("Meeting at 3pm tomorrow.") is False

    def test_rare_but_real_words_not_flagged(self):
        assert self.detector.predict("rhythms") is False
        assert self.detector.predict("strength") is False
        assert self.detector.predict("myths") is False

    def test_accented_text_not_flagged(self):
        assert self.detector.predict("café résumé naïve") is False

    def test_gibberish_flagged(self):
        assert self.detector.predict("xkqzv wpfjg mzxcv") is True
        assert self.detector.predict("asdkfj alskdjf") is True
        assert self.detector.predict("wqpzjx mvnbtr") is True

    def test_llr_sign_separates_classes(self):
        strategy = LogLikelihoodRatioStrategy()
        assert strategy._average_llr("the quick brown fox") > 0
        assert strategy._average_llr("xkqzv wpfjg mzxcv") < -1.0

    def test_abstains_on_too_few_bigrams(self):
        strategy = LogLikelihoodRatioStrategy()
        assert strategy.applicable("a") is False
        assert strategy.applicable("hello world") is True

    def test_empty_input(self):
        assert self.detector.predict("") is False
        assert self.detector.predict_proba("") == 0.0


class TestWordAnomalyStrategy:
    def setup_method(self):
        self.detector = GarbleDetector(Strategy.WORD_ANOMALY)

    def test_clean_english(self):
        assert self.detector.predict("order confirmed successfully") is False
        assert (
            self.detector.predict("She sells seashells by the seashore.")
            is False
        )

    def test_single_garbage_token_in_valid_sentence(self):
        # The headline use case: text-level averages dilute this away
        assert (
            self.detector.predict("order confirmed asdkjfhq thanks") is True
        )

    def test_all_gibberish(self):
        assert self.detector.predict("xkqzv wpfjg mzxcv") is True
        assert self.detector.predict_proba("xkqzv wpfjg mzxcv") == 1.0

    def test_rare_real_words_not_anomalous(self):
        strategy = WordAnomalyStrategy()
        for word in ("rhythms", "strength", "pneumonia", "sixths"):
            assert (
                strategy._word_log_prob(word)
                >= strategy.word_log_prob_threshold
            )

    def test_abstains_without_scoreable_words(self):
        strategy = WordAnomalyStrategy()
        assert strategy.applicable("a be to") is False
        assert strategy.applicable("hello world") is True

    def test_empty_input(self):
        assert self.detector.predict("") is False


class TestKeyboardAdjacencyStrategy:
    def setup_method(self):
        self.detector = GarbleDetector(Strategy.KEYBOARD_ADJACENCY)

    def test_keyboard_mash_flagged(self):
        assert self.detector.predict("asdfgh jklqwe") is True
        assert self.detector.predict("qwertyuiop") is True
        assert self.detector.predict("zxcvbnm asdfghjkl") is True

    def test_dictionary_words_exempt(self):
        # "typewriter" is entirely top-row; the dictionary exemption
        # must keep it clean
        assert self.detector.predict("typewriter repairs") is False
        assert self.detector.predict("sweater weather") is False

    def test_normal_text_not_flagged(self):
        assert self.detector.predict("the quick brown fox") is False
        assert self.detector.predict("Hello there, how are you?") is False

    def test_non_mash_gibberish_not_its_job(self):
        # Random consonants are not physical mash; other strategies
        # cover them
        assert self.detector.predict_proba("xkqzv wpfjg") == 0.0

    def test_empty_input(self):
        assert self.detector.predict("") is False


class TestEnsembleAbstention:
    def test_word_level_strategies_abstain_on_short_text(self):
        # An ensemble of only word-level strategies has no applicable
        # voter on short input and must return clean, not vote 0.0
        ensemble = EnsembleDetector(
            strategies=[Strategy.WORD_ANOMALY, Strategy.LOG_LIKELIHOOD_RATIO]
        )
        assert ensemble.predict("a") is False
        assert ensemble.predict_proba("a") == 0.0

    def test_applicable_detectors_still_vote(self):
        ensemble = EnsembleDetector(
            strategies=[
                Strategy.LOG_LIKELIHOOD_RATIO,
                Strategy.WORD_ANOMALY,
                Strategy.MARKOV_CHAIN,
            ]
        )
        assert ensemble.predict("xkqzv wpfjg mzxcv") is True
        assert ensemble.predict("the quick brown fox") is False

    def test_default_ensemble_still_works(self):
        ensemble = EnsembleDetector()
        assert ensemble.predict("Hello world") is False
        assert ensemble.predict("xkqzv wpfjg mzxcv") is True


class TestCoreContract:
    def test_batch_type_error_is_consistent(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, threads=4)
        with pytest.raises(TypeError):
            detector.predict(["hello", 123, "world"])
        big = ["hello"] * 12
        big[5] = None
        with pytest.raises(TypeError):
            detector.predict(big)

    def test_none_single_input_raises(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP)
        with pytest.raises(TypeError):
            detector.predict(None)

    def test_threshold_zero_empty_string_not_garbled(self):
        detector = GarbleDetector(Strategy.WORD_LOOKUP, threshold=0.0)
        assert detector.predict("") is False
        assert detector.predict("   ") is False

    def test_long_url_not_auto_flagged(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        url = "https://example.com/download/" + "a" * 1000
        assert detector.predict_proba(url) < 1.0
