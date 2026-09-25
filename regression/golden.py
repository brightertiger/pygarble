"""Frozen detector outputs that every implementation must reproduce.

Rows are JSONL: text, profile, garbled, score (12 decimals), status and
spans as [start, end, reason] with code-point offsets. Regenerate with
--write only when a behaviour change is intended; --check is run in CI.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import EnsembleDetector  # noqa: E402
from pygarble.ensemble import PROFILES  # noqa: E402
from regression.evaluate import challenge  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "golden.jsonl"
CHECKSUM = ROOT / "golden.sha256"

EDGE_INPUTS = [
    "",
    "   ",
    "a",
    "\U0001F44D\U0001F44F\U0001F64F",
    "你好世界",
    "Привет мир",
    "مرحبا بالعالم",
    "CafÃ© crÃ¨me",
    "result ��",
    "hello\x00world",
    "ab" * 2500,
    "https://example.com/path?q=1",
    "/usr/local/bin/python3",
    "4f8a9b2c1d3e5f6a7b8c9d0e",
    "aGVsbG8gd29ybGQgdGhpcyBpcw==",
    "123e4567-e89b-12d3-a456-426614174000",
    "don’t stop believin’",
    "Alice Nguyen and Priya Ramaswamy",
    "NASA FBI NATO UNESCO",
    "Order 000123 shipped 2026-09-26 at 10:00",
    "pаypal login",
    "café latté",
    "test test test",
    (
        "Sure. To rotate the logs, set the handler to RotatingFileHandler "
        "with a maximum size of ten megabytes and keep five backups. "
        "Restart the service afterwards and confirm that the new file is "
        "being written. If nothing appears, check the permissions first."
    ),
]


def texts():
    """Challenge texts then edge inputs, first occurrence wins."""
    seen = []
    for split in ("development", "holdout"):
        for case in challenge(split):
            seen.append(case["text"])
    seen.extend(EDGE_INPUTS)
    return list(dict.fromkeys(seen))


def rows():
    detectors = {
        name: EnsembleDetector(profile=name) for name in sorted(PROFILES)
    }
    for text in texts():
        for name, detector in detectors.items():
            analysis = detector.analyze(text)
            yield {
                "text": text,
                "profile": name,
                "garbled": analysis.garbled,
                "score": round(analysis.score, 12),
                "status": analysis.status,
                "spans": [[s.start, s.end, s.reason] for s in analysis.spans],
            }


def render():
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in rows()
    )


def check():
    expected = OUTPUT.read_text(encoding="utf-8")
    if hashlib.sha256(expected.encode("utf-8")).hexdigest() != (
        CHECKSUM.read_text().strip()
    ):
        print("golden.jsonl does not match golden.sha256", file=sys.stderr)
        return 1
    actual = render()
    if actual == expected:
        print(f"golden corpus reproduced ({actual.count(chr(10))} rows)")
        return 0
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    if len(expected_lines) != len(actual_lines):
        print(
            f"row count changed: {len(expected_lines)} committed, "
            f"{len(actual_lines)} generated",
            file=sys.stderr,
        )
    shown = 0
    for old, new in itertools.zip_longest(
        expected_lines, actual_lines, fillvalue="<missing>"
    ):
        if old != new:
            print(f"- {old}\n+ {new}", file=sys.stderr)
            shown += 1
            if shown == 10:
                break
    print("golden corpus differs; run --write if intended", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        content = render()
        OUTPUT.write_text(content, encoding="utf-8")
        CHECKSUM.write_text(
            hashlib.sha256(content.encode("utf-8")).hexdigest() + "\n"
        )
        print(f"wrote {content.count(chr(10))} rows")
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
