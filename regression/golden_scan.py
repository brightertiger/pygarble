"""Frozen Scanner outputs. Rows carry a text hash and findings, no text.

Inputs: every scan vector, every clean-corpus line, and the gibberish golden
inputs, scanned with all four categories at min_confidence 0.5 and
locales us,uk,in. A vector row's "options" are passed to Scanner as keyword
arguments and recorded in its golden row. sha256 is over the text encoded
as UTF-8 with surrogatepass, which equals plain UTF-8 for every valid string
and keeps the one lone-surrogate challenge text (WTF-8 bytes) hashable.
Regenerate with --write only for an intended behaviour change; --check runs
in CI.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import Scanner  # noqa: E402
from regression.golden import texts as gibberish_texts  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "golden_scan.jsonl"
CHECKSUM = ROOT / "golden_scan.sha256"
VECTORS = ROOT / "scan_vectors.json"
CORPUS = ROOT / "clean_corpus"


def texts():
    """(source, text, options) triples, first (text, options) wins."""
    seen = []
    for row in json.loads(VECTORS.read_text(encoding="utf-8")):
        seen.append(("vector", row["text"], row.get("options") or {}))
    for path in sorted(CORPUS.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            seen.append((path.name, line, {}))
    for text in gibberish_texts():
        seen.append(("golden", text, {}))
    unique = {}
    for source, text, options in seen:
        key = (text, json.dumps(options, sort_keys=True))
        unique.setdefault(key, (source, text, options))
    return list(unique.values())


def rows():
    scanners = {}
    for source, text, options in texts():
        key = json.dumps(options, sort_keys=True)
        if key not in scanners:
            scanners[key] = Scanner(**options)
        report = scanners[key].scan(text)
        yield {
            "source": source,
            "sha256": hashlib.sha256(
                text.encode("utf-8", "surrogatepass")
            ).hexdigest(),
            "length": report.length,
            "options": options,
            "flagged": report.flagged,
            "findings": [
                [
                    f.category,
                    f.kind,
                    f.start,
                    f.end,
                    round(f.confidence, 12),
                    f.reason,
                ]
                for f in report.findings
            ],
        }


def render():
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in rows()
    )


def check():
    for path in (OUTPUT, CHECKSUM):
        if not path.is_file():
            print(f"missing {path.name}; run --write first", file=sys.stderr)
            return 1
    expected = OUTPUT.read_text(encoding="utf-8")
    if hashlib.sha256(expected.encode("utf-8")).hexdigest() != (
        CHECKSUM.read_text().strip()
    ):
        print(
            "golden_scan.jsonl does not match golden_scan.sha256",
            file=sys.stderr,
        )
        return 1
    actual = render()
    if actual == expected:
        print(f"golden scan corpus reproduced ({actual.count(chr(10))} rows)")
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
    print(
        "golden scan corpus differs; run --write if intended", file=sys.stderr
    )
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
