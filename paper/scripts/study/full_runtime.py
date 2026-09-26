"""Isolated CPU benchmark for pygarble and the optional HF baseline."""

import argparse
import importlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .data import ROOT, documents, samples, write_json
from .detectors import make_model
from .full_corpus import PACKAGE_METHODS


def selected_inputs(cache: Path) -> list:
    primary = [r for r in samples(documents(cache)) if r["view"] == 400]
    positives = [r for r in primary if r["label"]]
    negatives = [r for r in primary if not r["label"]]
    return positives + [
        negatives[int(i * (len(negatives) - 1) / (len(positives) - 1))]
        for i in range(len(positives))
    ]


def measure(method: str, mode: str) -> dict:
    records = selected_inputs(ROOT / ".cache")
    texts = [r["text"] for r in records]
    begin = time.perf_counter_ns()
    if method == "hf_all":
        importlib.import_module("torch")
        importlib.import_module("transformers")
    else:
        importlib.import_module("pygarble")
    import_ns = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    if method == "hf_all":
        from .hf_backend import HFModel

        model = HFModel(ROOT / ".cache/hf-model", threads=1)
    else:
        model = make_model(method, [])
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
        "construction_or_load_ms": construction_ns / 1e6,
        "first_call_ms": first_ns / 1e6,
        "warm_median_ms": statistics.median(durations) / 1e6,
        "warm_p95_ms": ordered[int(0.95 * (len(ordered) - 1))] / 1e6,
        "warm_inputs_per_second": len(texts)
        / (statistics.median(elapsed) / 1e9),
        "inputs_per_pass": len(texts),
        "passes": 5,
        "characters_per_input": 400,
        "input_sha256": [r["text_sha256"] for r in records],
        "timed_path": "score_with_tokenization_if_hf",
        "threads": 1,
        "batch_size": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PACKAGE_METHODS + ("hf_all",))
    parser.add_argument("--mode", choices=("timing", "memory"))
    args = parser.parse_args()
    if args.method:
        print(json.dumps(measure(args.method, args.mode), allow_nan=False))
        return
    results = []
    for method in PACKAGE_METHODS + ("hf_all",):
        row = {"method": method}
        for mode in ("timing", "memory"):
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "paper.scripts.study.full_runtime",
                    "--method",
                    method,
                    "--mode",
                    mode,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            row.update(json.loads(child.stdout))
        results.append(row)
        print(method, row["warm_median_ms"], flush=True)
    write_json(ROOT / "full-results/runtime.json", results)


if __name__ == "__main__":
    main()
