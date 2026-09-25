"""LocalAnomaly: window configuration must be usable."""

import pytest

from pygarble import GarbleDetector, Strategy


def test_window_of_one_is_rejected():
    with pytest.raises(ValueError, match="window_words"):
        GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=1)


def test_window_span_is_emitted_for_dense_corruption():
    detector = GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=2)
    result = detector.analyze("hello xqzkvbwq qzxkvjwp world")
    reasons = {span.reason for span in result.spans}
    assert "corrupt_token_window" in reasons
