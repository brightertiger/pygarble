"""Threshold calibration: exact math on stub scores, one real run."""

import pytest

from pygarble import (
    CalibrationReport,
    EnsembleDetector,
    ThresholdPoint,
    calibrate,
)


class Stub:
    def __init__(self, table):
        self.table = table

    def score(self, texts):
        return [self.table[t] for t in texts]


def test_f1_recommends_gap_midpoint():
    stub = Stub({"g1": 0.9, "g2": 0.7, "c1": 0.6, "c2": 0.1})
    report = calibrate(stub, ["g1", "g2"], ["c1", "c2"])
    assert isinstance(report, CalibrationReport)
    assert report.objective == "f1"
    assert report.garbled == 2 and report.clean == 2
    assert report.recommended.threshold == pytest.approx(0.65)
    assert report.recommended.f1 == pytest.approx(1.0)
    assert report.recommended.false_positive_rate == 0.0
    assert [p.threshold for p in report.points] == pytest.approx(
        [0.0, 0.1, 0.6, 0.7, 0.9, 1.0]
    )


def test_point_math():
    stub = Stub({"g": 0.8, "c": 0.8})
    report = calibrate(stub, ["g"], ["c"], thresholds=[0.5, 0.9])
    low, high = report.points
    assert isinstance(low, ThresholdPoint)
    assert low.precision == pytest.approx(0.5)
    assert low.recall == 1.0 and low.false_positive_rate == 1.0
    assert high.precision == 1.0  # nothing flagged: precision defined as 1
    assert high.recall == 0.0 and high.f1 == 0.0


def test_max_fpr_objective_and_fallback():
    stub = Stub({"g1": 0.9, "g2": 0.4, "c1": 0.5, "c2": 0.1})
    report = calibrate(
        stub,
        ["g1", "g2"],
        ["c1", "c2"],
        objective="max_fpr",
        max_false_positive_rate=0.0,
    )
    assert report.recommended.threshold == pytest.approx(0.7)
    assert report.recommended.recall == pytest.approx(0.5)
    strict = calibrate(
        Stub({"g": 0.3, "c": 0.9}),
        ["g"],
        ["c"],
        objective="max_fpr",
        max_false_positive_rate=0.0,
    )
    assert strict.recommended.threshold == 1.0


def test_identical_scores_do_not_divide_by_zero():
    report = calibrate(Stub({"g": 0.0, "c": 0.0}), ["g"], ["c"])
    assert report.recommended.threshold in (0.0, 1.0)
    assert 0.0 <= report.recommended.f1 <= 1.0


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (dict(garbled=[], clean=["a"]), ValueError),
        (dict(garbled=["a"], clean=[]), ValueError),
        (dict(garbled=["a"], clean=["b"], objective="nope"), ValueError),
        (dict(garbled=["a"], clean=["b"], objective="max_fpr"), ValueError),
        (dict(garbled=[1], clean=["b"]), TypeError),
    ],
)
def test_validation(kwargs, error):
    with pytest.raises(error):
        calibrate(EnsembleDetector(), **kwargs)


def test_real_detector_round_trip():
    garbled = ["qxzjkwpv bnmqwer", "asdfghjkl", "xkrf plmq bvzt nwsd"]
    clean = ["hello world", "the quick brown fox", "please send the invoice"]
    report = calibrate(EnsembleDetector(), garbled, clean)
    assert report.recommended.recall == 1.0
    assert report.recommended.false_positive_rate == 0.0
    assert report.recommended.threshold < 1.0
    detector = EnsembleDetector(threshold=report.recommended.threshold)
    assert detector.predict(garbled) == [True, True, True]
    assert detector.predict(clean) == [False, False, False]


def test_max_fpr_fallback_is_not_moved_to_a_midpoint():
    # A clean text scoring exactly 1.0 makes 1.0 an observed score; the
    # unmet-constraint fallback must still recommend the 1.0 cut itself.
    report = calibrate(
        Stub({"g": 0.3, "c": 1.0}),
        ["g"],
        ["c"],
        objective="max_fpr",
        max_false_positive_rate=0.0,
    )
    assert report.recommended.threshold == 1.0
    assert report.recommended.false_positive_rate == 1.0


def test_max_false_positive_rate_rejected_under_f1():
    with pytest.raises(ValueError, match="requires objective='max_fpr'"):
        calibrate(
            Stub({"g": 0.9, "c": 0.1}),
            ["g"],
            ["c"],
            max_false_positive_rate=0.1,
        )
