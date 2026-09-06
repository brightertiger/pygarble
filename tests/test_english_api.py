"""Public contracts and independent regressions for English detection."""

import dataclasses
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import TimeoutError
from pathlib import Path

import pytest

from pygarble import Analysis, EnsembleDetector, GarbleDetector, Strategy
from pygarble.analysis import Evidence
from pygarble.preprocessing import TextFeatures


@pytest.mark.parametrize(
    "text", ["नमस्ते दुनिया", "你好世界", "Привет мир", "مرحبا بالعالم"]
)
def test_non_english_is_expected_gibberish(text):
    result = EnsembleDetector().analyze(text)
    assert result.garbled
    assert result.score > 0.9
    assert result.signals[0].reason == "outside_english_alphabet"
    # Specialist encoding checks do not invent English-language evidence.
    assert not EnsembleDetector(profile="corruption").predict(text)


@pytest.mark.parametrize("threads", [None, 4])
@pytest.mark.parametrize(
    "method", ["predict", "predict_proba", "analyze", "score"]
)
@pytest.mark.parametrize("bad", [None, 0, False, 123, {}, b"hello"])
def test_batch_types_validated_before_any_work(threads, method, bad):
    detector = EnsembleDetector(threads=threads)
    calls = []
    detector._detectors[0]._strategy_instance.evaluate = (
        lambda features: calls.append(features)
    )
    with pytest.raises(TypeError, match="element 10"):
        getattr(detector, method)(["hello"] * 10 + [bad])
    assert calls == []


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "1"]
)
def test_finite_weight_contract(value):
    with pytest.raises(ValueError, match="weights"):
        EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN],
            voting="weighted",
            weights=[value],
        )


@pytest.mark.parametrize("threads", [0, -1, 1.5, True, float("nan")])
def test_worker_count_contract(threads):
    for factory in [
        EnsembleDetector,
        lambda **kw: GarbleDetector(Strategy.MARKOV_CHAIN, **kw),
    ]:
        with pytest.raises(ValueError, match="threads"):
            factory(threads=threads)


def test_large_finite_weights_do_not_overflow():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN, Strategy.LOG_LIKELIHOOD_RATIO],
        voting="weighted",
        weights=[1e308, 1e308],
    )
    assert 0.9 < detector.score("qxzjkwpv") <= 1


def test_member_options_do_not_collide_with_detector_threshold():
    detector = EnsembleDetector(
        strategies=[Strategy.BIGRAM_PROBABILITY, Strategy.WORD_ANOMALY],
        threshold=0.8,
        strategy_kwargs={
            Strategy.BIGRAM_PROBABILITY: {"threshold": 0.2},
            Strategy.WORD_ANOMALY: {"min_word_length": 9},
        },
    )
    assert detector.threshold == 0.8
    assert detector._detectors[0]._strategy_instance.threshold == 0.2
    assert detector._detectors[1]._strategy_instance.min_word_length == 9
    with pytest.raises(ValueError, match="not selected"):
        EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN],
            strategy_kwargs={Strategy.WORD_LOOKUP: {}},
        )


def test_unknown_settings_warn():
    with pytest.warns(DeprecationWarning, match="min_lenght"):
        GarbleDetector(Strategy.MARKOV_CHAIN, min_lenght=8)


@pytest.mark.parametrize(
    "strategy,options",
    [
        (Strategy.WORD_ANOMALY, {"min_word_length": 0}),
        (Strategy.LOG_LIKELIHOOD_RATIO, {"llr_scale": -1}),
        (Strategy.KEYBOARD_ADJACENCY, {"chain_threshold": 2.5}),
        (Strategy.WORD_LOOKUP, {"unknown_threshold": float("nan")}),
        (Strategy.LETTER_FREQUENCY, {"deviation_threshold": 0}),
        (Strategy.BIGRAM_PROBABILITY, {"threshold": float("nan")}),
    ],
)
def test_strategy_numeric_validation(strategy, options):
    with pytest.raises(ValueError):
        GarbleDetector(strategy, strategy_kwargs=options)


def test_word_lookup_threshold_changes_decision():
    text = "hello qxzjkwpv"
    assert GarbleDetector(Strategy.WORD_LOOKUP, unknown_threshold=0.2).predict(
        text
    )
    assert not GarbleDetector(
        Strategy.WORD_LOOKUP, unknown_threshold=0.8
    ).predict(text)


def test_length_is_not_universal_gibberish_evidence():
    for strategy in [
        Strategy.MOJIBAKE,
        Strategy.UNICODE_SCRIPT,
        Strategy.CONTROL_CHARACTERS,
    ]:
        assert not GarbleDetector(strategy).predict("a" * 2000)
    with pytest.raises(ValueError, match="max_input_length"):
        EnsembleDetector(max_input_length=10).predict(["valid", "a" * 11])
    # The opt-in legacy policy remains available.
    assert GarbleDetector(Strategy.MOJIBAKE, max_string_length=10).predict(
        "a" * 11
    )


@pytest.mark.parametrize(
    "voting", ["any", "all", "average", "weighted", "majority"]
)
def test_analysis_scalar_batch_and_threads_agree(voting):
    texts = [
        "hello world",
        "",
        "qxzjkwpv",
        "CafÃ©",
        "नमस्ते दुनिया",
        "qwerty",
    ] * 3
    options = {"voting": voting}
    if voting == "weighted":
        options["weights"] = [1.0] * len(EnsembleDetector().strategies)
    serial = EnsembleDetector(**options)
    threaded = EnsembleDetector(threads=3, **options)
    analyses = serial.analyze(texts)
    assert threaded.analyze(texts) == analyses
    assert serial.predict(texts) == [result.garbled for result in analyses]
    assert serial.score(texts) == [result.score for result in analyses]
    for text, result in zip(texts, analyses):
        assert serial.analyze(text) == result
        assert isinstance(result, Analysis)
        json.dumps(dataclasses.asdict(result))


def test_majority_votes_are_distinct_from_mean_score():
    detector = EnsembleDetector(
        strategies=[
            Strategy.MARKOV_CHAIN,
            Strategy.WORD_ANOMALY,
            Strategy.LOG_LIKELIHOOD_RATIO,
        ],
        voting="majority",
    )
    for member, value in zip(detector._detectors, [0.49, 0.49, 1.0]):
        member._strategy_instance.evaluate = (
            lambda features, score=value: Evidence(score)
        )
    result = detector.analyze("hello")
    assert not result.garbled
    assert result.score > 0.5


def test_zero_weight_only_applicable_member_abstains():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
        voting="weighted",
        weights=[0, 1],
        threshold=0,
    )
    assert not detector.predict("hi")
    assert detector.analyze("hi").status == "insufficient_evidence"


def test_domain_allowlist_does_not_hide_encoding_artifacts():
    detector = EnsembleDetector(
        allowlist=(word for word in ["syzygy", "qxzjkwpv"])
    )
    assert not detector.predict("syzygy qxzjkwpv")
    assert detector.predict("syzygy qxzjkwpv \x00")
    with pytest.raises(TypeError):
        EnsembleDetector(allowlist="qxzjkwpv")


def test_folded_tokens_preserve_original_offsets():
    text = "Cafe\u0301, ﬁle: qxzjkwpvm!"
    features = TextFeatures(text)
    assert [(token.text, token.folded) for token in features.tokens] == [
        ("Cafe\u0301", "cafe"),
        ("ﬁle", "file"),
        ("qxzjkwpvm", "qxzjkwpvm"),
    ]
    for token in features.tokens:
        assert text[token.start : token.end] == token.text
    result = GarbleDetector(Strategy.LOCAL_ANOMALY).analyze(text)
    assert any(
        text[span.start : span.end] == "qxzjkwpvm" for span in result.spans
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hello\x00world",
        "bad\x07message",
        "text\ud800here",
        "broken � record",
        "x" + "\u0301" * 12,
    ],
)
def test_raw_corruption_with_spans(text):
    result = GarbleDetector(Strategy.CONTROL_CHARACTERS).analyze(text)
    assert result.garbled and result.spans
    assert all(
        0 <= span.start < span.end <= len(text) for span in result.spans
    )


@pytest.mark.parametrize(
    "text", ["hello\tworld\nnext\rline", "Café", "👩‍💻", "a\u0301", "नमस्ते"]
)
def test_control_detector_preserves_legitimate_unicode(text):
    assert not GarbleDetector(Strategy.CONTROL_CHARACTERS).predict(text)


def test_local_corruption_is_not_diluted_by_a_paragraph():
    text = (
        "The order has been reviewed and approved for delivery. " * 20
        + "qxzjkwpvm"
        + " Please contact us if you need help." * 20
    )
    assert GarbleDetector(Strategy.LOCAL_ANOMALY).predict(text)


@pytest.mark.parametrize(
    "layout,text",
    [("qwerty", "qwerty12345"), ("azerty", "azerty"), ("qwertz", "qwertz")],
)
def test_configurable_keyboard_layouts(layout, text):
    assert GarbleDetector(
        Strategy.KEYBOARD_ADJACENCY, keyboard_layout=layout
    ).predict(text)
    assert not GarbleDetector(
        Strategy.KEYBOARD_ADJACENCY, keyboard_layout=layout
    ).predict("typewriter power")


def test_bounded_phrase_cycles():
    detector = GarbleDetector(Strategy.REPETITION)
    assert detector.predict("red blue green " * 4)
    assert not detector.predict("red blue green orange purple yellow")


def test_timeout_propagates_instead_of_returning_clean():
    detector = GarbleDetector(
        Strategy.MARKOV_CHAIN, threads=2, timeout_per_text=0.001
    )
    event = threading.Event()

    def slow(features):
        event.wait(0.05)
        return Evidence(0.9)

    detector._strategy_instance.evaluate = slow
    with pytest.raises(TimeoutError):
        detector.predict(["qxzjkwpv"] * 10)


def test_lazy_import_does_not_load_dictionary_for_specialist():
    program = (
        "import sys; from pygarble import GarbleDetector, Strate"
        "gy; GarbleDetector(Strategy.CONTROL_CHARACTERS).predict"
        "('hello'); assert 'pygarble.data.words' not in sys.modu"
        "les"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_decisions_are_deterministic_across_processes():
    program = (
        "import dataclasses,json; from pygarble import EnsembleD"
        "etector; print(json.dumps([dataclasses.asdict(x) for x "
        "in EnsembleDetector().analyze(['hello','qxzjkwpv','नमस्"
        "ते दुनिया','CafÃ©'])],sort_keys=True))"
    )
    results = [
        subprocess.check_output(
            [sys.executable, "-c", program],
            env=dict(os.environ, PYTHONHASHSEED=seed),
        )
        for seed in ["1", "932"]
    ]
    assert results[0] == results[1]


def test_frozen_challenge_has_no_family_leakage():
    from regression.evaluate import challenge

    assert challenge("development")
    assert challenge("holdout")


def test_model_manifest_matches_packaged_data():
    import hashlib

    import pygarble.data

    directory = Path(pygarble.data.__file__).parent
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            == checksum
        )


def test_word_lookup_upper_boundary_maps_to_decision_score():
    detector = GarbleDetector(Strategy.WORD_LOOKUP, unknown_threshold=1.0)
    assert detector.score("qxzjkwpv") == 0.5
    assert detector.predict("qxzjkwpv")
    assert detector.score("hello world") == 0.0
