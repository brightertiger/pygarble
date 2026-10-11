"""pygarble calibrate() on B dev, then realised FPR and capture elsewhere.

500 tweets and 500 random strings are drawn from B dev; calibrate() picks
the highest-recall threshold with FPR at most 1 percent on them; the
profile is rebuilt with that threshold and applied to B test and A.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from . import data, metrics
from .run import write_json

PROFILES = ("english", "english_fusion")
SAMPLE = 500
TARGET = 0.01


def draw(rows: Sequence[data.Row]) -> Dict[str, List[str]]:
    rng = random.Random(data.SEED)
    tweets = [row.text for row in rows if row.label == 0]
    junk = [row.text for row in rows if row.label == 1]
    clean = rng.sample(tweets, SAMPLE)
    return {"clean": clean, "garbled": rng.sample(junk, SAMPLE)}


def calibrate_profile(
    profile: str, sample: Dict[str, List[str]]
) -> Dict[str, Any]:
    from pygarble import EnsembleDetector, calibrate

    report = calibrate(
        EnsembleDetector(profile=profile, threads=1),
        sample["garbled"],
        sample["clean"],
        objective="max_fpr",
        max_false_positive_rate=TARGET,
    )
    point = report.recommended
    return {
        "threshold": point.threshold,
        "dev_fpr": round(point.false_positive_rate, metrics.DIGITS),
        "dev_recall": round(point.recall, metrics.DIGITS),
        # calibrate() falls back to 1.0 when no cut meets the target.
        "constraint_met": point.false_positive_rate <= TARGET,
        "detector": EnsembleDetector(
            profile=profile, threshold=point.threshold, threads=1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data.DEFAULT_ROOT)
    args = parser.parse_args()
    rows = data.load(args.data_root)
    sample = draw(rows["B_dev"])
    test = rows["B_test"]
    labels = np.array([row.label == 1 for row in test])
    idx_a, idx_b = metrics.bootstrap_indices(len(rows["A"]), len(test))
    out = {}
    for profile in PROFILES:
        result = calibrate_profile(profile, sample)
        detector = result.pop("detector")
        a_flag = np.array(detector.predict([row.text for row in rows["A"]]))
        b_flag = np.array(detector.predict([row.text for row in test]))
        boot_flag, boot_labels = b_flag[idx_b], labels[idx_b]
        result.update(
            {
                "B_test_fpr": metrics.estimate(
                    float(b_flag[~labels].mean()),
                    metrics.ratio(
                        (boot_flag & ~boot_labels).sum(axis=1),
                        (~boot_labels).sum(axis=1),
                    ),
                ),
                "B_test_recall": metrics.estimate(
                    float(b_flag[labels].mean()),
                    metrics.ratio(
                        (boot_flag & boot_labels).sum(axis=1),
                        boot_labels.sum(axis=1),
                    ),
                ),
                "A_capture": metrics.estimate(
                    float(a_flag.mean()), a_flag[idx_a].mean(axis=1)
                ),
            }
        )
        out[profile] = result
        print(profile, json.dumps(result), flush=True)
    write_json(
        metrics.RESULTS / "calibration.json",
        {
            "target_fpr": TARGET,
            "sample": {"clean": SAMPLE, "garbled": SAMPLE, "seed": data.SEED},
            "profiles": out,
        },
    )
    from .report import write_report

    write_report(json.loads((metrics.RESULTS / "metrics.json").read_text()))


if __name__ == "__main__":
    main()
