"""Recompute metrics.json and pr_curves.json from predictions and manifest.

Exits non-zero on any mismatch. Needs numpy but neither the raw data nor
the baselines.
"""

import json
import sys
from typing import Any, List

from . import data, metrics


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
    curves = json.loads((metrics.RESULTS / "pr_curves.json").read_text())
    gate = json.loads((metrics.RESULTS / "checkpoint.json").read_text())
    families = {n: m["family"] for n, m in saved["methods"].items()}
    files = {
        path.stem
        for part in metrics.SETS
        for path in (metrics.PREDICTIONS / part).glob("*.csv")
    }
    problems = []
    if files != set(families):
        problems.append("prediction files and metrics.json methods differ")
    fresh, fresh_curves = metrics.compute(
        manifest, metrics.load_all(manifest, list(families)), families
    )
    fresh, fresh_curves = json.loads(json.dumps(fresh)), json.loads(
        json.dumps(fresh_curves)
    )
    problems += differences(saved, fresh, "metrics")
    problems += differences(curves, fresh_curves, "pr_curves")
    gate.pop("override", None)
    problems += differences(gate, fresh["checkpoint"], "checkpoint")
    return problems


def main() -> int:
    problems = check()
    for problem in problems[:50]:
        print(problem)
    if problems:
        print(f"verify: {len(problems)} mismatches")
        return 1
    print("verify: metrics.json and pr_curves.json match the predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
