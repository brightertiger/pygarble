"""llm_output profile: degenerate model output, quiet on technical prose."""

import pytest

from pygarble import EnsembleDetector
from pygarble.ensemble import PROFILES

PARAGRAPH = (
    "Sure. To rotate the logs, set the handler to RotatingFileHandler with "
    "a maximum size of ten megabytes and keep five backups. Restart the "
    "service afterwards and confirm that the new file is being written. "
    "If nothing appears, check the directory permissions first."
)


def test_profile_members():
    assert [s.value for s in PROFILES["llm_output"]] == [
        "repetition",
        "control_characters",
        "mojibake",
        "local_anomaly",
    ]
    assert EnsembleDetector(profile="llm_output").voting == "any"


@pytest.mark.parametrize(
    "text",
    [
        "the the the the the the the the",
        "and so on and so on and so on and so on and so on",
        "I am happy to help! I am happy to help! I am happy to help! "
        "I am happy to help! I am happy to help!",
        "The cafÃ© was closed",
        "Result: ���",
        "hello\x00world",
        "Please review the qxzkvbwq qzxkvjwp xkqzvbwr output carefully",
    ],
)
def test_degenerate_output_is_flagged(text):
    assert EnsembleDetector(profile="llm_output").predict(text) is True


@pytest.mark.parametrize(
    "text",
    [
        PARAGRAPH,
        'def parse(row):\n    return row.split(",")',
        # A text made entirely of one repeated token ("x = x + x + x + x")
        # is flagged by design, so this case uses varied identifiers.
        "total = price * qty + tax",
        "Use Kubernetes with Prometheus and Grafana on GKE.",
        'The API returned 200 OK with ETag W/"abc123".',
    ],
)
def test_technical_prose_is_clean(text):
    assert EnsembleDetector(profile="llm_output").predict(text) is False
