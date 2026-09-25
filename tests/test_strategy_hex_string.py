"""HexString: paths and identifiers are not base64."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "/usr/local/bin/python3",
        "src/components/HeaderView",
        "getUserAccountBalanceV2",
        "some/path/withDigits1/x",
    ],
)
def test_paths_and_identifiers_are_clean(text):
    assert GarbleDetector(Strategy.HEX_STRING).score(text) < 0.5


@pytest.mark.parametrize(
    "text",
    [
        "aGVsbG8gd29ybGQgdGhpcyBpcw==",
        "U29tZSByYW5kb20gYmFzZTY0IHN0cmluZw",
        "4f8a9b2c1d3e5f6a7b8c9d0e",
    ],
)
def test_real_base64_and_hex_still_fire(text):
    assert GarbleDetector(Strategy.HEX_STRING).score(text) >= 0.5
