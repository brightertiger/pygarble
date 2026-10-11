"""Recompute metrics.json and tuning/tuning.json from saved files.

metrics.json comes from the saved predictions and the manifest;
tuning.json from the cached strategy scores, the recorded tuning row ids
and the learned weights (Optuna is not re-run).

Exits non-zero on any mismatch. Needs numpy but neither the raw data nor
the baselines.
"""

import json
import sys
from typing import Any, List

from . import data, metrics, tune


def differences(saved: Any, fresh: Any, path: str = "") -> List[str]:
    if isinstance(saved, dict) and isinstance(fresh, dict):
        out = []
        for key in sorted(set(saved) | set(fresh)):
            if key not in saved or key not in fresh:
                out.append(f"{path}/{key}: present in one side only")
            else:
                out += differences(saved[key], fresh[key], f"{path}/{key}")
        return out
    if isinstance(saved, list) and isinstance(fresh, list):
        if len(saved) != len(fresh):
            return [f"{path}: length {len(saved)} != {len(fresh)}"]
        out = []
        for i, (a, b) in enumerate(zip(saved, fresh)):
            out += differences(a, b, f"{path}[{i}]")
        return out
    return [] if saved == fresh else [f"{path}: {saved!r} != {fresh!r}"]


def check() -> List[str]:
    manifest = json.loads(data.MANIFEST.read_text())
    saved = json.loads((metrics.RESULTS / "metrics.json").read_text())
    families = {n: m["family"] for n, m in saved["methods"].items()}
    problems = []
    for part in metrics.SETS:
        files = {p.stem for p in (metrics.PREDICTIONS / part).glob("*.csv")}
        if files != set(families):
            problems.append(f"{part} prediction files and methods differ")
    stray = {p.name for p in metrics.PREDICTIONS.iterdir()} - set(metrics.SETS)
    if stray:
        problems.append(f"unexpected prediction folders: {sorted(stray)}")
    fresh = metrics.compute(
        manifest, metrics.load_all(manifest, list(families)), families
    )
    problems += differences(saved, json.loads(json.dumps(fresh)), "metrics")
    tuned = json.loads(tune.RESULT.read_text())
    fresh_tuned = json.loads(json.dumps(tune.recompute()))
    problems += differences(tuned, fresh_tuned, "tuning")
    return problems


def main() -> int:
    problems = check()
    for problem in problems[:50]:
        print(problem)
    if problems:
        print(f"verify: {len(problems)} mismatches")
        return 1
    print("verify: metrics.json and tuning.json match the saved files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
