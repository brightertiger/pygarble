"""Throughput of Scanner per category on a synthetic corpus. Numbers are
published in the README with the machine noted; nothing is promised."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import Scanner  # noqa: E402
from pygarble.findings import CATEGORIES  # noqa: E402

PARAGRAPHS = [
    "The quarterly report covers revenue, churn and the hiring plan for the "
    "next two quarters. Please read the summary before the meeting.",
    "To rotate the logs, set the handler to RotatingFileHandler with a "
    "maximum size of ten megabytes and keep five backups.",
    "Customers can update their shipping address from the account page. "
    "Changes apply to orders that have not yet been dispatched.",
    "def parse(row): return row.split(',')  # naive CSV split used in tests",
    "GET /api/v1/users/42 200 12ms request_id=7f3a9c",
]
PLANTED = [
    "contact jane.doe@example.com for details",
    "card 4111 1111 1111 1111 on file",
    "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    "what the fuck happened here",
    "qxzjkwpv bnmqwer zzxqv",
]


def build_corpus(size_bytes: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []
    total = 0
    while total < size_bytes:
        line = rng.choice(PARAGRAPHS)
        if rng.random() < 0.05:
            line = line + " " + rng.choice(PLANTED)
        lines.append(line)
        total += len(line.encode("utf-8")) + 1
    return lines


def measure(categories: List[str], lines: List[str]) -> Dict[str, float]:
    scanner = Scanner(categories=categories)
    size = sum(len(line.encode("utf-8")) + 1 for line in lines)
    findings = 0
    start = time.perf_counter()
    for line in lines:
        findings += len(scanner.scan(line).findings)
    elapsed = time.perf_counter() - start
    return {
        "mb_per_s": (size / 1e6) / elapsed if elapsed else float("inf"),
        "lines_per_s": len(lines) / elapsed if elapsed else float("inf"),
        "findings": findings,
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    lines = build_corpus(int(args.size_mb * 1e6))
    results = {}
    for category in CATEGORIES:
        results[category] = measure([category], lines)
    results["all"] = measure(list(CATEGORIES), lines)
    results["rules_only"] = measure(["secrets", "pii", "profanity"], lines)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    print(f"{'category':<12}{'MB/s':>10}{'lines/s':>12}{'findings':>10}")
    for name, row in results.items():
        print(
            f"{name:<12}{row['mb_per_s']:>10.1f}{row['lines_per_s']:>12.0f}"
            f"{row['findings']:>10}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
