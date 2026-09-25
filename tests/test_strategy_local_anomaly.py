"""LocalAnomaly: window configuration must be usable."""

from pygarble import GarbleDetector, Strategy


def test_window_of_one_flags_a_single_bad_word():
    detector = GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=1)
    result = detector.analyze("please send the xqzkvbwp report today")
    reasons = {span.reason for span in result.spans}
    assert result.garbled is True
    assert result.score == 0.8
    assert "severe_local_anomaly" in reasons
    assert "corrupt_token_window" in reasons


def test_window_span_is_emitted_for_dense_corruption():
    detector = GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=2)
    result = detector.analyze("hello xqzkvbwq qzxkvjwp world")
    reasons = {span.reason for span in result.spans}
    assert "corrupt_token_window" in reasons
