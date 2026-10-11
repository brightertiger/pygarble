"""Tune a weight per strategy and a threshold on a small labelled sample.

Every strategy is scored once on every row of A and B with default
settings and cached (no text). For each of five seeds a stratified 20
percent of A and of B is the tuning sample; Optuna's TPE sampler searches
a weight in [0, 1] per strategy (0 excludes it) and a threshold in
[0.2, 0.8] for the best F1 on that sample, under pygarble's weighted
voting. The tuned configuration, rebuilt as a real EnsembleDetector, is
evaluated on the held-out 80 percent next to the english and
english_fusion profiles on the same rows. Run from the repository root:

    python -m paper.scripts.benchmark.tune --data-root <pygarble-data>

`verify.py` recomputes tuning.json from the cached scores, the recorded
row ids and the learned parameters, without Optuna or the raw data.
"""

import argparse
import csv
import gzip
import io
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import data, metrics

TUNING = metrics.RESULTS / "tuning"
SCORES = TUNING / "strategy_scores.csv.gz"
WEIGHTS = TUNING / "weights.json"
SPLITS = TUNING / "splits.json"
RESULT = TUNING / "tuning.json"
REPORT = TUNING / "tuning.md"
SEEDS = tuple(data.SEED + k for k in range(5))
PRIMARY = SEEDS[0]
FRACTION = 0.2
TRIALS = 400
WEIGHT_STEP = 0.05
THRESHOLDS = (0.2, 0.8)
THRESHOLD_STEP = 0.01
PROFILES = ("english", "english_fusion")
METHODS = PROFILES + ("tuned",)
METRICS = ("A_capture", "B_precision", "B_recall", "B_fpr")
Matrix = Tuple[np.ndarray, np.ndarray]  # scores, applicable (rows x 32)


def strategy_names() -> List[str]:
    from pygarble import Strategy

    return [strategy.value for strategy in Strategy]


# --- cached strategy scores -------------------------------------------


def score_rows(
    rows: Sequence[data.Row], names: Sequence[str], log: bool = False
) -> List:
    """(row id, strategy, score, applicable); inapplicable scores are 0."""
    from pygarble import GarbleDetector

    texts = [row.text for row in rows]
    out = []
    for name in names:
        started = time.perf_counter()
        analyses = GarbleDetector(name, threads=1).analyze(texts)
        for row, analysis in zip(rows, analyses):
            signal = analysis.signals[0]
            score = float(signal.score) if signal.applicable else 0.0
            out.append((row.id, name, score, bool(signal.applicable)))
        if log:
            seconds = time.perf_counter() - started
            print(f"  {name}: {seconds:.1f}s", flush=True)
    return out


def write_scores(records: Sequence) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("row_id", "strategy", "score", "applicable"))
    for row_id, name, score, applicable in records:
        writer.writerow((row_id, name, repr(score), int(applicable)))
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    with SCORES.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(buffer.getvalue().encode())


def read_scores(manifest: Dict, names: Sequence[str]) -> Dict[str, Matrix]:
    """Score and applicability matrices for A and B in manifest order."""
    column = {name: j for j, name in enumerate(names)}
    out = {}
    with gzip.open(SCORES, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for part in metrics.SETS:
        ids = manifest[f"{part}_ids"]
        position = {row_id: i for i, row_id in enumerate(ids)}
        scores = np.full((len(ids), len(names)), np.nan)
        flags = np.zeros((len(ids), len(names)), dtype=bool)
        for row in rows:
            i = position.get(row["row_id"])
            if i is None:
                continue
            j = column[row["strategy"]]
            scores[i, j] = float(row["score"])
            flags[i, j] = row["applicable"] == "1"
        if np.isnan(scores).any():
            raise ValueError(f"{SCORES}: missing scores for {part}")
        out[part] = (scores, flags)
    return out


# --- weighted voting, as EnsembleDetector computes it -----------------


class Sum:
    """Row-wise float sum in the order and arithmetic of Python's sum().

    Python 3.12 sums floats with Neumaier compensation; adding 0.0 for a
    dropped member leaves both the sum and the compensation unchanged.
    """

    def __init__(self, rows: int) -> None:
        self.total = np.zeros(rows)
        self.carry = np.zeros(rows)

    def add(self, x: np.ndarray) -> None:
        t = self.total + x
        big = np.abs(self.total) >= np.abs(x)
        self.carry = self.carry + np.where(
            big, (self.total - t) + x, (x - t) + self.total
        )
        self.total = t

    def value(self) -> np.ndarray:
        fix = (self.carry != 0) & np.isfinite(self.carry)
        return np.where(fix, self.total + self.carry, self.total)


def weighted(
    matrix: Matrix, weights: Sequence[float], threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Scores and decisions of EnsembleDetector(voting="weighted").

    Members that are inapplicable or weighted 0 are dropped; the score is
    the weighted mean of the rest, with weights divided by the largest
    remaining weight and summed in member order, exactly as
    `EnsembleDetector._aggregate` does on Python 3.12, so scores agree bit
    for bit. A row with no remaining member scores 0 and is clean.
    """
    scores, applicable = matrix
    w = np.asarray(weights, dtype=float)
    active = applicable & (w > 0)
    scale = np.where(active, w, 0.0).max(axis=1)
    any_member = scale > 0
    safe = np.where(any_member, scale, 1.0)
    total, numerator = Sum(len(scores)), Sum(len(scores))
    for j in np.flatnonzero(w > 0):
        member = active[:, j]
        total.add(np.where(member, w[j] / safe, 0.0))
        numerator.add(np.where(member, scores[:, j] * w[j] / safe, 0.0))
    denominator = np.where(any_member, total.value(), 1.0)
    score = np.where(any_member, numerator.value() / denominator, 0.0)
    return score, any_member & (score >= threshold)


def f1(flags: np.ndarray, labels: np.ndarray) -> float:
    tp = int((flags & labels).sum())
    wrong = int((flags != labels).sum())
    return 2 * tp / (2 * tp + wrong) if tp else 0.0


# --- tuning samples ---------------------------------------------------


def draw(manifest: Dict, seed: int) -> Dict[str, List[str]]:
    """Stratified 20 percent of A and of B (by label), in manifest order."""
    rng = np.random.default_rng(seed)
    a_ids = manifest["A_ids"]
    chosen_a = rng.choice(
        len(a_ids), round(FRACTION * len(a_ids)), replace=False
    )
    b_ids = manifest["B_ids"]
    chosen_b: List[int] = []
    for label in (0, 1):
        pool = [
            i
            for i, row_id in enumerate(b_ids)
            if manifest["B_labels"][row_id] == label
        ]
        picked = rng.choice(len(pool), round(FRACTION * len(pool)), False)
        chosen_b += [pool[i] for i in picked]
    return {
        "A": [a_ids[i] for i in sorted(chosen_a)],
        "B": [b_ids[i] for i in sorted(chosen_b)],
    }


def masks(manifest: Dict, sample: Dict[str, List[str]]) -> Dict:
    return {
        part: np.isin(manifest[f"{part}_ids"], sample[part])
        for part in metrics.SETS
    }


def pooled(
    manifest: Dict, matrices: Dict[str, Matrix], keep: Dict
) -> Tuple[Matrix, np.ndarray]:
    """Rows of A and B under `keep`, stacked, with junk labels."""
    b_labels = np.array(
        [manifest["B_labels"][i] == 1 for i in manifest["B_ids"]]
    )
    scores = np.vstack([matrices[p][0][keep[p]] for p in metrics.SETS])
    flags = np.vstack([matrices[p][1][keep[p]] for p in metrics.SETS])
    labels = np.concatenate(
        [np.ones(int(keep["A"].sum()), dtype=bool), b_labels[keep["B"]]]
    )
    return (scores, flags), labels


def search(
    matrix: Matrix, labels: np.ndarray, names: Sequence[str], seed: int
) -> Dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        weights = [
            trial.suggest_float(name, 0.0, 1.0, step=WEIGHT_STEP)
            for name in names
        ]
        threshold = trial.suggest_float(
            "threshold", *THRESHOLDS, step=THRESHOLD_STEP
        )
        if not any(weights):
            return 0.0
        return f1(weighted(matrix, weights, threshold)[1], labels)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=TRIALS)
    seconds = time.perf_counter() - started
    best = study.best_trial
    return {
        "seed": seed,
        "threshold": round(best.params["threshold"], 2),
        "weights": {name: round(best.params[name], 2) for name in names},
        "best_trial": best.number,
        "seconds": round(seconds, 2),
    }


# --- held-out evaluation ----------------------------------------------


def heldout_metrics(
    manifest: Dict,
    matrices: Dict[str, Matrix],
    profiles: Dict[str, Dict[str, metrics.Prediction]],
    params: Dict[str, Any],
    sample: Dict[str, List[str]],
) -> Dict[str, Any]:
    keep = {p: ~m for p, m in masks(manifest, sample).items()}
    names = list(params["weights"])
    weights = [params["weights"][n] for n in names]
    held = {
        "A_ids": [i for i, k in zip(manifest["A_ids"], keep["A"]) if k],
        "B_ids": [i for i, k in zip(manifest["B_ids"], keep["B"]) if k],
        "B_labels": manifest["B_labels"],
    }
    predictions = {
        name: {p: (sets[p][0][keep[p]], sets[p][1][keep[p]]) for p in sets}
        for name, sets in profiles.items()
    }
    predictions["tuned"] = {
        part: weighted(
            (matrices[part][0][keep[part]], matrices[part][1][keep[part]]),
            weights,
            params["threshold"],
        )
        for part in metrics.SETS
    }
    families = {name: "pygarble" for name in profiles}
    families["tuned"] = "tuned"
    return metrics.compute(held, predictions, families)


def compute(
    manifest: Dict,
    matrices: Dict[str, Matrix],
    profiles: Dict[str, Dict[str, metrics.Prediction]],
    draws: Sequence[Dict[str, Any]],
    samples: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """tuning.json contents; a function of saved files only."""
    per_seed = []
    primary: Dict[str, Any] = {}
    for params in draws:
        sample = samples[str(params["seed"])]
        names = list(params["weights"])
        matrix, labels = pooled(manifest, matrices, masks(manifest, sample))
        tuning_f1 = f1(
            weighted(
                matrix,
                [params["weights"][n] for n in names],
                params["threshold"],
            )[1],
            labels,
        )
        result = heldout_metrics(manifest, matrices, profiles, params, sample)
        if params["seed"] == PRIMARY:
            primary = result
        per_seed.append(
            {
                "seed": params["seed"],
                "tuning_f1": round(tuning_f1, metrics.DIGITS),
                "rows": result["rows"],
                "methods": {
                    name: {
                        key: result["methods"][name][key]["value"]
                        for key in METRICS
                    }
                    for name in METHODS
                },
            }
        )
    spread = {
        name: {
            key: {
                "mean": round(float(np.mean(values)), metrics.DIGITS),
                "sd": round(float(np.std(values, ddof=1)), metrics.DIGITS),
            }
            for key in METRICS
            for values in [[s["methods"][name][key] for s in per_seed]]
        }
        for name in METHODS
    }
    return {
        "evaluation": "held-out 80 percent of A and of B per seed",
        "fraction_tuning": FRACTION,
        "trials": TRIALS,
        "seeds": [params["seed"] for params in draws],
        "primary_seed": PRIMARY,
        "resamples": metrics.RESAMPLES,
        "bootstrap_seed": data.SEED,
        "primary": {
            "rows": primary["rows"],
            "methods": {
                name: {key: primary["methods"][name][key] for key in METRICS}
                for name in METHODS
            },
        },
        "per_seed": per_seed,
        "mean_sd": spread,
    }


def load_saved() -> Tuple[Dict, Dict, Dict, List, Dict]:
    manifest = json.loads(data.MANIFEST.read_text())
    weights = json.loads(WEIGHTS.read_text())
    samples = json.loads(SPLITS.read_text())["tuning_ids"]
    names = weights["strategies"]
    matrices = read_scores(manifest, names)
    profiles = metrics.load_all(manifest, PROFILES)
    return manifest, matrices, profiles, weights["draws"], samples


def recompute() -> Dict[str, Any]:
    manifest, matrices, profiles, draws, samples = load_saved()
    for params in draws:
        if draw(manifest, params["seed"]) != samples[str(params["seed"])]:
            raise ValueError(f"seed {params['seed']}: sample differs")
    return compute(manifest, matrices, profiles, draws, samples)


# --- the real ensemble ------------------------------------------------


def ensemble(params: Dict[str, Any]) -> Any:
    from pygarble import EnsembleDetector

    chosen = [n for n, w in params["weights"].items() if w > 0]
    return EnsembleDetector(
        strategies=chosen,
        weights=[params["weights"][n] for n in chosen],
        voting="weighted",
        threshold=params["threshold"],
        threads=1,
    )


def check_ensemble(
    manifest: Dict,
    matrices: Dict[str, Matrix],
    rows: Dict[str, List[data.Row]],
    params: Dict[str, Any],
    sample: Dict[str, List[str]],
) -> int:
    """Re-score held-out rows with EnsembleDetector; decisions must match."""
    detector = ensemble(params)
    names = list(params["weights"])
    weights = [params["weights"][n] for n in names]
    checked = 0
    for part, chosen in masks(manifest, sample).items():
        keep = ~chosen
        texts = [row.text for row, k in zip(rows[part], keep) if k]
        real = np.array(detector.predict(texts))
        cached = weighted(
            (matrices[part][0][keep], matrices[part][1][keep]),
            weights,
            params["threshold"],
        )[1]
        if not np.array_equal(real, cached):
            bad = int((real != cached).sum())
            raise AssertionError(f"seed {params['seed']} {part}: {bad} differ")
        checked += len(texts)
    return checked


# --- reports ----------------------------------------------------------


def cell(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_report(result: Dict[str, Any], weights: Dict[str, Any]) -> None:
    labels = {
        "english": "pygarble (english)",
        "english_fusion": "pygarble (english_fusion)",
        "tuned": "pygarble (tuned weights)",
    }
    rows = result["primary"]["rows"]
    lines = [
        "# Weight tuning study",
        "",
        f"Tuning sample: a stratified {FRACTION:.0%} of A and of B per seed;"
        f" Optuna {weights['optuna']} TPE, {TRIALS} trials, objective F1.",
        f"Held-out rows (primary seed {PRIMARY}): A {rows['A']}, B "
        f"{rows['B']} ({rows['B_positive']} random strings, "
        f"{rows['B_negative']} tweets).",
        "",
        f"## Held-out metrics, primary seed {PRIMARY} (95% bootstrap CI)",
        "",
        "| method | A capture | B precision | B recall | B FPR |",
        "|---|---|---|---|---|",
    ]
    for name in METHODS:
        entry = result["primary"]["methods"][name]
        cells = []
        for key in METRICS:
            low, high = entry[key]["ci"] or (None, None)
            cells.append(
                f"{cell(entry[key]['value'])} [{cell(low)}, {cell(high)}]"
            )
        lines.append(f"| {labels[name]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Held-out metrics, mean ± sd over {len(SEEDS)} seeds",
        "",
        "| method | A capture | B precision | B recall | B FPR |",
        "|---|---|---|---|---|",
    ]
    for name in METHODS:
        spread = result["mean_sd"][name]
        lines.append(
            f"| {labels[name]} | "
            + " | ".join(
                f"{spread[k]['mean']:.3f} ± {spread[k]['sd']:.3f}"
                for k in METRICS
            )
            + " |"
        )
    lines += [
        "",
        "## Learned parameters",
        "",
        "| seed | threshold | members | tuning F1 | best trial | seconds |",
        "|---|---|---|---|---|---|",
    ]
    f1s = {s["seed"]: s["tuning_f1"] for s in result["per_seed"]}
    for params in weights["draws"]:
        members = sum(w > 0 for w in params["weights"].values())
        lines.append(
            f"| {params['seed']} | {params['threshold']:.2f} | {members} | "
            f"{f1s[params['seed']]:.3f} | {params['best_trial']} | "
            f"{params['seconds']:.1f} |"
        )
    lines += [
        "",
        f"Total tuning time {weights['seconds_total']:.1f} s; scoring the "
        f"cache took {weights['scoring_seconds']:.1f} s.",
        "",
        "| strategy | mean weight | primary-seed weight |",
        "|---|---|---|",
    ]
    primary = next(d for d in weights["draws"] if d["seed"] == PRIMARY)
    for name, mean in sorted(
        weights["mean_weight"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(
            f"| {name} | {mean:.2f} | {primary['weights'][name]:.2f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument(
        "--rescore", action="store_true", help="rebuild the score cache"
    )
    args = parser.parse_args()
    manifest = json.loads(data.MANIFEST.read_text())
    rows = data.load(args.data_root)
    names = strategy_names()
    scoring_seconds = None
    if args.rescore or not SCORES.exists():
        started = time.perf_counter()
        records = []
        for part in metrics.SETS:
            print(f"score: {part}", flush=True)
            records += score_rows(rows[part], names, log=True)
        write_scores(records)
        scoring_seconds = round(time.perf_counter() - started, 1)
    elif WEIGHTS.exists():
        scoring_seconds = json.loads(WEIGHTS.read_text())["scoring_seconds"]
    matrices = read_scores(manifest, names)
    samples = {str(seed): draw(manifest, seed) for seed in SEEDS}
    draws = []
    for seed in SEEDS:
        matrix, labels = pooled(
            manifest, matrices, masks(manifest, samples[str(seed)])
        )
        params = search(matrix, labels, names, seed)
        params["heldout_rows_checked"] = check_ensemble(
            manifest, matrices, rows, params, samples[str(seed)]
        )
        print(
            f"seed {seed}: threshold {params['threshold']}, "
            f"{params['seconds']}s",
            flush=True,
        )
        draws.append(params)
    weights = {
        "optuna": metadata.version("optuna"),
        "sampler": "TPESampler(seed=<draw seed>)",
        "trials": TRIALS,
        "weight_step": WEIGHT_STEP,
        "threshold_range": list(THRESHOLDS),
        "threshold_step": THRESHOLD_STEP,
        "voting": "weighted",
        "strategies": names,
        "scoring_seconds": scoring_seconds,
        "seconds_total": round(sum(d["seconds"] for d in draws), 2),
        "draws": draws,
        "mean_weight": {
            name: round(float(np.mean([d["weights"][name] for d in draws])), 3)
            for name in names
        },
    }
    write_json(WEIGHTS, weights)
    write_json(SPLITS, {"fraction": FRACTION, "tuning_ids": samples})
    profiles = metrics.load_all(manifest, PROFILES)
    result = compute(manifest, matrices, profiles, draws, samples)
    write_json(RESULT, result)
    write_report(result, weights)
    print(f"wrote {RESULT.relative_to(data.REPOSITORY)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
