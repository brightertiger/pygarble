"""Operating points, metrics and bootstrap intervals from saved predictions.

Shared by run.py and verify.py so every reported number is a function of
`results/predictions/` and `manifest.json` alone.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import OUTPUT, SEED

RESULTS = OUTPUT / "results"
PREDICTIONS = RESULTS / "predictions"
SETS = ("A", "B_dev", "B_test")
TARGETS = {"fpr1": 0.01, "fpr0.1": 0.001}
POINTS = ("native",) + tuple(TARGETS)
RESAMPLES = 1000
CHECKPOINT = ("english_fusion", "fpr1", 0.70)
PR_STEP = 0.005
DIGITS = 6
Prediction = Tuple[np.ndarray, np.ndarray]  # scores, native decisions


def rounded_score(value: float) -> float:
    """Scores are saved and analysed at ten significant digits."""
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


def cut_for_fpr(negative_scores: np.ndarray, target: float) -> float:
    """(k+1)-th highest negative score; flag scores strictly above it."""
    k = math.floor(target * len(negative_scores) + 1e-9)
    if k >= len(negative_scores):
        return -math.inf
    return float(np.sort(negative_scores)[::-1][k])


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


def auroc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray):
    """Mann-Whitney AUROC, ties one half, per row of a weight matrix."""
    _, inverse = np.unique(scores, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1]
    pos = np.add.reduceat((weights * labels)[:, order], starts, axis=1)
    neg = np.add.reduceat((weights * ~labels)[:, order], starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    u = (pos * (below + 0.5 * neg)).sum(axis=1)
    return ratio(u, pos.sum(axis=1) * neg.sum(axis=1))


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> List[List[float]]:
    """[threshold, precision, recall] for `score >= threshold`, thinned."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    points: List[List[float]] = []
    for i in last:
        recall = tp[i] / labels.sum()
        if points and recall - points[-1][2] < PR_STEP and i != last[-1]:
            continue
        points.append(
            [
                round(float(sorted_scores[i]), DIGITS),
                round(float(tp[i] / (tp[i] + fp[i])), DIGITS),
                round(float(recall), DIGITS),
            ]
        )
    return points


def load_all(
    manifest: Dict, methods: Sequence[str], sets: Sequence[str] = SETS
) -> Dict[str, Dict[str, Prediction]]:
    ids = {
        "A": manifest["A_ids"],
        "B_dev": manifest["B_dev_ids"],
        "B_test": manifest["B_test_ids"],
    }
    return {
        name: {
            part: read_predictions(
                PREDICTIONS / part / f"{name}.csv", ids[part]
            )
            for part in sets
        }
        for name in methods
    }


def fit_cuts(
    manifest: Dict, dev: Dict[str, Prediction]
) -> Dict[str, Dict[str, Any]]:
    labels = np.array([manifest["B_labels"][i] for i in manifest["B_dev_ids"]])
    out = {}
    for name, (scores, _) in dev.items():
        negatives = scores[labels == 0]
        cuts = {p: cut_for_fpr(negatives, t) for p, t in TARGETS.items()}
        out[name] = {
            "cut": cuts,
            "dev_fpr": {
                p: round(float(np.mean(negatives > c)), DIGITS)
                for p, c in cuts.items()
            },
            # False when ties at the top leave the cut at the highest B dev
            # score, so it flags no B dev row; A or B test rows above the
            # cut are still flagged and counted.
            "reachable": {p: bool(c < scores.max()) for p, c in cuts.items()},
        }
    return out


def checkpoint(cuts: Dict, a_scores: np.ndarray) -> Dict[str, Any]:
    name, point, floor = CHECKPOINT
    capture = float(np.mean(a_scores > cuts[name]["cut"][point]))
    return {
        "method": name,
        "point": point,
        "floor": floor,
        "A_capture": round(capture, DIGITS),
        "passed": capture >= floor,
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
) -> Tuple[Dict, Dict]:
    """metrics.json and pr_curves.json contents."""
    cuts = fit_cuts(manifest, {n: p["B_dev"] for n, p in predictions.items()})
    labels = np.array(
        [manifest["B_labels"][i] == 1 for i in manifest["B_test_ids"]]
    )
    n_a, n_b = len(manifest["A_ids"]), len(labels)
    idx_a, idx_b = bootstrap_indices(n_a, n_b)
    weights = np.stack([np.bincount(row, minlength=n_b) for row in idx_b])
    boot_labels = labels[idx_b]
    methods: Dict[str, Any] = {}
    curves: Dict[str, Any] = {}
    for name, sets in predictions.items():
        a_scores, a_native = sets["A"]
        b_scores, b_native = sets["B_test"]
        entry: Dict[str, Any] = {
            "family": families[name],
            "auroc": estimate(
                float(auroc(b_scores, labels, np.ones((1, n_b)))[0]),
                auroc(b_scores, labels, weights),
            ),
            "points": {},
        }
        for point in POINTS:
            if point == "native":
                a_flag, b_flag = a_native, b_native
            else:
                cut = cuts[name]["cut"][point]
                a_flag, b_flag = a_scores > cut, b_scores > cut
            boot_flag = b_flag[idx_b]
            tp = (boot_flag & boot_labels).sum(axis=1)
            fp = (boot_flag & ~boot_labels).sum(axis=1)
            positives = boot_labels.sum(axis=1)
            full_tp = int((b_flag & labels).sum())
            full_fp = int((b_flag & ~labels).sum())
            found: Dict[str, Any] = {}
            if point != "native":
                found = {
                    key: cuts[name][key][point]
                    for key in ("cut", "dev_fpr", "reachable")
                }
            entry["points"][point] = found
            found.update(
                {
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
                    "B_recall": estimate(
                        full_tp / labels.sum(), ratio(tp, positives)
                    ),
                    "B_fpr": estimate(
                        full_fp / (~labels).sum(), ratio(fp, n_b - positives)
                    ),
                }
            )
        methods[name] = entry
        curves[name] = pr_curve(b_scores, labels)
    first = next(iter(predictions.values()))
    metrics = {
        "checkpoint": (
            checkpoint(cuts, predictions[CHECKPOINT[0]]["A"][0])
            if CHECKPOINT[0] in predictions
            else None
        ),
        "resamples": RESAMPLES,
        "seed": SEED,
        "rows": {"A": len(first["A"][0]), "B_test": n_b},
        "methods": methods,
    }
    return metrics, curves
