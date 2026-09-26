"""Isolated benchmark worker; timing and RSS use separate processes."""

import argparse
import importlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

from .data import documents, samples
from .detectors import make_model


def measure(cache: Path, method: str, mode: str) -> dict:
    docs = documents(cache)
    primary = [r for r in samples(docs) if r["view"] == 400]
    positives = [r for r in primary if r["label"]]
    negatives = [r for r in primary if not r["label"]]
    selected = positives + [
        negatives[int(i * (len(negatives) - 1) / (len(positives) - 1))]
        for i in range(len(positives))
    ]
    texts = [r["text"] for r in selected]
    training = [d["text"][:100000] for d in docs if d["family"] == "secreta"]
    begin = time.perf_counter_ns()
    if method not in ("keep_all", "char_bigram", "char_trigram"):
        importlib.import_module("pygarble")
    import_ns = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    model = make_model(method, training)
    construction_ns = time.perf_counter_ns() - begin
    if mode == "memory":
        for text in texts:
            model.score(text)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_process_rss_bytes": (
                rss if platform.system() == "Darwin" else rss * 1024
            )
        }
    begin = time.perf_counter_ns()
    model.score(texts[0])
    first_ns = time.perf_counter_ns() - begin
    # One unmeasured warm pass precedes all five measured passes.
    for text in texts:
        model.score(text)
    durations = []
    elapsed = []
    for _ in range(5):
        batch_start = time.perf_counter_ns()
        for text in texts:
            begin = time.perf_counter_ns()
            model.score(text)
            durations.append(time.perf_counter_ns() - begin)
        elapsed.append(time.perf_counter_ns() - batch_start)
    ordered = sorted(durations)
    return {
        "import_ms": import_ns / 1e6,
        "construction_or_training_ms": construction_ns / 1e6,
        "first_call_ms": first_ns / 1e6,
        "warm_median_ms": statistics.median(durations) / 1e6,
        "warm_p95_ms": ordered[int(0.95 * (len(ordered) - 1))] / 1e6,
        "warm_documents_per_second": len(texts)
        / (statistics.median(elapsed) / 1e9),
        "warm_utf8_bytes_per_second": sum(len(t.encode()) for t in texts)
        / (statistics.median(elapsed) / 1e9),
        "documents_per_pass": len(texts),
        "passes": 5,
        "characters_per_document": 400,
        "timed_path": "score_and_applicability",
        "threads": 1,
        "training_family_for_character_models": "secreta",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", choices=("memory", "timing"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            measure(args.cache, args.method, args.mode), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
