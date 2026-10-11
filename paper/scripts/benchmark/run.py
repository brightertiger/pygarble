"""Score every method on A and B dev, apply the checkpoint, then B test once.

Writes per-row predictions (row id, score, decision; no text) and the
metrics computed from them. Run from the repository root:

    python -m paper.scripts.benchmark.run --data-root <pygarble-data>
"""

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import data, metrics
from .methods import BASELINE_FILES, specs

ENVIRONMENT = metrics.RESULTS / "environment.json"
DISTRIBUTIONS = (
    "pygarble",
    "numpy",
    "nostril",
    "gibberish-detector",
    "gibberish-chat-detector",
    "lingua-language-detector",
)


def check_baseline_files(root: Path) -> None:
    for name, expected in BASELINE_FILES.items():
        actual = data.sha256(root / "baselines" / name)
        if actual != expected:
            raise ValueError(f"baselines/{name}: sha256 {actual}")


def score_set(
    name: str, method, part: str, rows: Sequence[data.Row]
) -> np.ndarray:
    scores, flags = method.run([row.text for row in rows])
    scores = [metrics.rounded_score(s) for s in scores]
    metrics.write_predictions(
        metrics.PREDICTIONS / part / f"{name}.csv",
        [row.id for row in rows],
        scores,
        flags,
    )
    return np.array(scores)


def git(*args: str) -> str:
    return subprocess.run(
        ("git",) + args,
        cwd=data.REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def cpu() -> str:
    try:
        return subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def environment() -> Dict:
    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu(),
        "python": sys.version.split()[0],
        "packages": versions,
        "pygarble_commit": git("rev-parse", "HEAD"),
        "pygarble_tree_clean": git("status", "--porcelain", "pygarble") == "",
        "nostril_commit": "fbc0c91249283a9fbc9036206391ce1138826fd3",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def save_metrics(names: List[str]) -> Dict:
    manifest = json.loads(data.MANIFEST.read_text())
    every = specs()
    predictions = metrics.load_all(manifest, names)
    result, curves = metrics.compute(
        manifest, predictions, {n: every[n].family for n in names}
    )
    write_json(metrics.RESULTS / "metrics.json", result)
    write_json(metrics.RESULTS / "pr_curves.json", curves)
    from .report import write_report

    write_report(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="recompute metrics from saved predictions without scoring",
    )
    args = parser.parse_args()
    every = specs(args.data_root)
    names = list(every)
    if args.metrics_only:
        save_metrics(names)
        return 0
    test_dir = metrics.PREDICTIONS / "B_test"
    if test_dir.exists() and any(test_dir.iterdir()):
        print("B test was already scored; refusing a second test run.")
        return 1
    check_baseline_files(args.data_root)
    rows = data.load(args.data_root)
    manifest = json.loads(data.MANIFEST.read_text())
    built = {}
    dev_scores = {}
    a_scores = {}
    for name in names:
        print(f"dev: {name}", flush=True)
        built[name] = every[name].build()
        a_scores[name] = score_set(name, built[name], "A", rows["A"])
        dev_scores[name] = (
            score_set(name, built[name], "B_dev", rows["B_dev"]),
            None,
        )
    cuts = metrics.fit_cuts(manifest, dev_scores)
    gate = metrics.checkpoint(cuts, a_scores[metrics.CHECKPOINT[0]])
    write_json(metrics.RESULTS / "checkpoint.json", gate)
    print(f"checkpoint: {gate}", flush=True)
    if not gate["passed"]:
        print("Checkpoint failed; B test not scored.")
        return 2
    for name in names:
        print(f"test: {name}", flush=True)
        score_set(name, built[name], "B_test", rows["B_test"])
    write_json(ENVIRONMENT, environment())
    save_metrics(names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
