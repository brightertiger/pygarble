"""Local timing and memory observations, excluded from CI gates."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import EnsembleDetector, __version__
from pygarble.strategies.repetition import RepetitionStrategy
from regression.benchmark import load_test_cases


def median_time(function):
    return statistics.median(timeit.repeat(function, number=1, repeat=5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    texts = [
        case["text"]
        for case in load_test_cases(
            str(Path(__file__).parent / "benchmark_data.json")
        )
    ]
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "samples": len(texts),
        "notes": (
            "Median of five warm runs; local observations, not portable "
            "service-level guarantees. Import measurements include "
            "tracemalloc instrumentation."
        ),
        "batches_seconds": {},
        "unique_word_repetition_seconds": {},
    }
    for profile in ["legacy", "english", "english_extended"]:
        for workers in [None, 4]:
            detector = EnsembleDetector(profile=profile, threads=workers)
            detector.predict(texts)
            report["batches_seconds"][f"{profile}_threads_{workers}"] = (
                median_time(lambda: detector.predict(texts))
            )
    repetition = RepetitionStrategy()
    for size in [1000, 2000, 4000]:
        text = " ".join("word" + str(i) for i in range(size))
        report["unique_word_repetition_seconds"][str(size)] = median_time(
            lambda: repetition._check_word_repetition(text)
        )
    code = (
        "import json,time,tracemalloc,sys; tracemalloc.start(); "
        "start=time.perf_counter(); "
        "from pygarble import GarbleDetector,Strategy; "
        "GarbleDetector(Strategy.CONTROL_CHARACTERS).predict('hello'); "
        "print(json.dumps(dict(seconds=time.perf_counter()-start, "
        "peak_bytes=tracemalloc.get_traced_memory()[1], "
        "dictionary_loaded='pygarble.data.words' in sys.modules)))"
    )
    report["specialist_import"] = json.loads(
        subprocess.check_output([sys.executable, "-c", code])
    )
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
