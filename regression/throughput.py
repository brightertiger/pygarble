"""Throughput of Scanner per category on a synthetic corpus.

The corpus is synthetic: five short ASCII English paragraphs (prose, a log
line and a line of code) chosen at random with a fixed seed, with a finding
(email, card, AWS key, profanity or gibberish) planted in about 5% of lines.
By default each line is scanned on its own (about 110 bytes per call, so
per-call overhead dominates); with --chunk-bytes N the lines are joined
with newlines into documents of about N bytes, so MB/s reflects per-byte
cost. MB is 10^6 UTF-8 bytes of scanned text. Numbers are published in the
README with the machine noted; nothing is promised.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def _chunk(lines: List[str], chunk_bytes: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for line in lines:
        current.append(line)
        size += len(line.encode("utf-8")) + 1
        if size >= chunk_bytes:
            chunks.append("\n".join(current))
            current, size = [], 0
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_corpus(
    size_bytes: int, seed: int = 7, chunk_bytes: int = 0
) -> List[str]:
    """Seeded lines totalling about size_bytes; with chunk_bytes > 0 the
    same lines joined by newlines into texts of about chunk_bytes."""
    rng = random.Random(seed)
    lines: List[str] = []
    total = 0
    while total < size_bytes:
        line = rng.choice(PARAGRAPHS)
        if rng.random() < 0.05:
            line = line + " " + rng.choice(PLANTED)
        lines.append(line)
        total += len(line.encode("utf-8")) + 1
    return _chunk(lines, chunk_bytes) if chunk_bytes > 0 else lines


def measure(categories: List[str], texts: List[str]) -> Dict[str, Any]:
    """Scan each text once; MB is 10^6 UTF-8 bytes of scanned text.
    "lines_per_s" counts texts, which are documents with --chunk-bytes."""
    scanner = Scanner(categories=categories)
    size = sum(len(text.encode("utf-8")) for text in texts)
    findings = 0
    start = time.perf_counter()
    for text in texts:
        findings += len(scanner.scan(text).findings)
    elapsed = time.perf_counter() - start
    return {
        "mb_per_s": (size / 1e6) / elapsed if elapsed else 0.0,
        "lines_per_s": len(texts) / elapsed if elapsed else 0.0,
        "findings": findings,
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=float, default=10.0)
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=0,
        help="join lines into texts of about N bytes (0: one line per text)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    texts = build_corpus(
        int(args.size_mb * 1e6), chunk_bytes=max(0, args.chunk_bytes)
    )
    results: Dict[str, Any] = {}
    for category in CATEGORIES:
        results[category] = measure([category], texts)
    results["all"] = measure(list(CATEGORIES), texts)
    results["rules_only"] = measure(["secrets", "pii", "profanity"], texts)
    if args.json:
        results["chunk_bytes"] = max(0, args.chunk_bytes)
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    print(f"chunk_bytes={max(0, args.chunk_bytes)} texts={len(texts)}")
    print(f"{'category':<12}{'MB/s':>10}{'texts/s':>12}{'findings':>10}")
    for name, row in results.items():
        print(
            f"{name:<12}{row['mb_per_s']:>10.1f}{row['lines_per_s']:>12.0f}"
            f"{row['findings']:>10}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
