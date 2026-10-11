"""Metrics and bootstrap intervals from saved predictions.

Shared by run.py and verify.py so every reported number is a function of
`results/predictions/` and `manifest.json` alone. Every method is scored at
its own default decision; no thresholds are fitted.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import OUTPUT, SEED

RESULTS = OUTPUT / "results"
PREDICTIONS = RESULTS / "predictions"
SETS = ("A", "B")
RESAMPLES = 1000
DIGITS = 6
Prediction = Tuple[np.ndarray, np.ndarray]  # scores, default decisions


def rounded_score(value: float) -> float:
    """Scores are saved at ten significant digits."""
    return float(f"{value:.10g}")


def write_predictions(
    path: Path, ids: Sequence[str], scores: Sequence[float], flags: Sequence
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("row_id", "score", "decision"))
        for row_id, score, flag in zip(ids, scores, flags):
            writer.writerow((row_id, f"{score:.10g}", int(bool(flag))))


def read_predictions(path: Path, ids: Sequence[str]) -> Prediction:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["row_id"] for row in rows] != list(ids):
        raise ValueError(f"{path}: row ids do not match the manifest")
    return (
        np.array([float(row["score"]) for row in rows]),
        np.array([row["decision"] == "1" for row in rows]),
    )


def interval(values: np.ndarray) -> Optional[List[float]]:
    values = values[~np.isnan(values)]
    if not len(values):
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), DIGITS), round(float(high), DIGITS)]


def estimate(point: float, samples: np.ndarray) -> Dict[str, Any]:
    value = None if math.isnan(point) else round(point, DIGITS)
    return {"value": value, "ci": interval(samples)}


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def load_all(
    manifest: Dict, methods: Sequence[str]
) -> Dict[str, Dict[str, Prediction]]:
    ids = {"A": manifest["A_ids"], "B": manifest["B_ids"]}
    return {
        name: {
            part: read_predictions(
                PREDICTIONS / part / f"{name}.csv", ids[part]
            )
            for part in SETS
        }
        for name in methods
    }


def bootstrap_indices(n_a: int, n_b: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return (
        rng.integers(0, n_a, size=(RESAMPLES, n_a)),
        rng.integers(0, n_b, size=(RESAMPLES, n_b)),
    )


def compute(
    manifest: Dict,
    predictions: Dict[str, Dict[str, Prediction]],
    families: Dict[str, str],
) -> Dict[str, Any]:
    """metrics.json contents: default-decision metrics for every method."""
    labels = np.array(
        [manifest["B_labels"][i] == 1 for i in manifest["B_ids"]]
    )
    n_a, n_b = len(manifest["A_ids"]), len(labels)
    idx_a, idx_b = bootstrap_indices(n_a, n_b)
    boot_labels = labels[idx_b]
    positives = boot_labels.sum(axis=1)
    methods: Dict[str, Any] = {}
    for name, sets in predictions.items():
        a_flag = sets["A"][1]
        b_flag = sets["B"][1]
        boot_flag = b_flag[idx_b]
        tp = (boot_flag & boot_labels).sum(axis=1)
        fp = (boot_flag & ~boot_labels).sum(axis=1)
        full_tp = int((b_flag & labels).sum())
        full_fp = int((b_flag & ~labels).sum())
        methods[name] = {
            "family": families[name],
            "A_capture": estimate(
                float(a_flag.mean()), a_flag[idx_a].mean(axis=1)
            ),
            "B_precision": estimate(
                (
                    full_tp / (full_tp + full_fp)
                    if full_tp + full_fp
                    else math.nan
                ),
                ratio(tp, tp + fp),
            ),
            "B_recall": estimate(full_tp / labels.sum(), ratio(tp, positives)),
            "B_fpr": estimate(
                full_fp / (~labels).sum(), ratio(fp, n_b - positives)
            ),
        }
    return {
        "decision": "default",
        "resamples": RESAMPLES,
        "seed": SEED,
        "rows": {
            "A": n_a,
            "B": n_b,
            "B_positive": int(labels.sum()),
            "B_negative": int((~labels).sum()),
        },
        "methods": methods,
    }
