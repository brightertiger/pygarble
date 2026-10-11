"""Pronounceability and calibrate stay fast on large inputs, same results."""

import random
import time

from pygarble import GarbleDetector, calibrate
from pygarble.gibberish.calibration import ThresholdPoint


def test_pronounceability_is_linear_on_long_consonant_run_and_unchanged():
    # F1: scores pinned at full repr before the fix.
    detector = GarbleDetector("pronounceability")
    pinned = {
        "CSFXbon": "0.9849999999999999",
        "BDRßTNDmv": "0.99",
        "mnbvcxz": "0.99",
        "Rhythms and strengths in myths": "0.0",
        "asdkjh qwpoeiru zmxncbv": "1.0",
    }
    for text, score in pinned.items():
        assert repr(detector.predict_proba(text)) == score, text
    start = time.perf_counter()
    assert detector.predict_proba("xkcdqwrtzpl" * 100_000) == 1.0
    # The quadratic coda took about 10 s here; linear takes well under 1 s.
    assert time.perf_counter() - start < 3.0


class _Scores:
    """Scores texts by label; rounding makes scores tie, as real ones do."""

    def __init__(self, seed, n, digits=None):
        rng = random.Random(seed)
        self.by_label = {
            label: [
                rng.random() if digits is None else round(rng.random(), digits)
                for _ in range(n)
            ]
            for label in ("g", "c")
        }

    def score(self, texts):
        return list(self.by_label[texts[0]])


def _reference_points(garbled, clean, candidates):
    """The pre-fix O(N^2) sweep, kept verbatim as the oracle."""
    points = []
    for t in candidates:
        tp = sum(1 for s in garbled if s >= t)
        fp = sum(1 for s in clean if s >= t)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / len(garbled)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        points.append(
            ThresholdPoint(t, precision, recall, f1, fp / len(clean))
        )
    return tuple(points)


def test_calibrate_matches_reference_sweep_and_scales():
    # F2: identical points at full repr on 2000+2000 texts, and fast at 20k.
    detector = _Scores(1, 2000, digits=3)
    garbled_scores = detector.by_label["g"]
    clean_scores = detector.by_label["c"]
    report = calibrate(detector, ["g"] * 2000, ["c"] * 2000)
    candidates = sorted(set(garbled_scores) | set(clean_scores) | {0.0, 1.0})
    expected = _reference_points(garbled_scores, clean_scores, candidates)
    assert repr(report.points) == repr(expected)
    start = time.perf_counter()
    calibrate(_Scores(3, 20000), ["g"] * 20000, ["c"] * 20000)
    # Distinct scores: the quadratic sweep took about 30 s here.
    assert time.perf_counter() - start < 5.0
