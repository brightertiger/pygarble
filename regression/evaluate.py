"""Reproducible English evaluation; legacy labels are never overwritten."""

import argparse
import hashlib
import json
import platform
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import EnsembleDetector, GarbleDetector, Strategy, __version__
from regression.benchmark import load_test_cases

ROOT = Path(__file__).resolve().parent


def reviewed_legacy():
    cases = load_test_cases(str(ROOT / "benchmark_data.json"))
    overrides = json.loads((ROOT / "label_overrides.json").read_text())
    unique = {}
    for case in cases:
        if case["category"] in overrides["clean_categories"]:
            case["expected"] = False
        if case["category"] in overrides["garbled_categories"]:
            case["expected"] = True
        case["expected"] = overrides["text_overrides"].get(
            case["text"], case["expected"]
        )
        if (
            case["text"] in unique
            and unique[case["text"]]["expected"] != case["expected"]
        ):
            raise ValueError("conflicting reviewed labels")
        unique[case["text"]] = case
    return list(unique.values())


def challenge(split):
    path = ROOT / "english_challenge.json"
    expected = (ROOT / "english_challenge.sha256").read_text().strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(
            "challenge checksum changed; version the dataset explicitly"
        )
    cases = json.loads(path.read_text())["cases"]
    families = defaultdict(set)
    identities = set()
    for case in cases:
        if case["id"] in identities:
            raise ValueError("duplicate challenge identity")
        identities.add(case["id"])
        families[case["family"]].add(case["split"])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("family leaks across challenge splits")
    return [
        dict(
            text=c["text"],
            expected=c["expected_garbled"],
            category=c["family"],
            source=c["source"],
        )
        for c in cases
        if c["split"] == split
    ]


def metrics(cases, predictions, applicable=None):
    tp = sum(bool(p) and c["expected"] for c, p in zip(cases, predictions))
    fp = sum(bool(p) and not c["expected"] for c, p in zip(cases, predictions))
    positives = sum(c["expected"] for c in cases)
    negatives = len(cases) - positives
    result = {
        "n": len(cases),
        "tp": tp,
        "fp": fp,
        "tn": negatives - fp,
        "fn": positives - tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
    }
    if applicable is not None:
        result["coverage"] = sum(applicable) / len(cases) if cases else None
    # Wilson 95% FPR interval, including zero observed false positives.
    if negatives:
        z = 1.959963984540054
        rate = fp / negatives
        denominator = 1 + z * z / negatives
        center = (rate + z * z / (2 * negatives)) / denominator
        half = (
            z
            * (
                rate * (1 - rate) / negatives
                + z * z / (4 * negatives * negatives)
            )
            ** 0.5
            / denominator
        )
        result["fpr_95_interval"] = [
            max(0.0, center - half),
            min(1.0, center + half),
        ]
    return result


def evaluate(detector, cases):
    analyses = [detector.analyze(c["text"]) for c in cases]
    predictions = [a.garbled for a in analyses]
    result = metrics(
        cases,
        predictions,
        [a.status != "insufficient_evidence" for a in analyses],
    )
    groups = {}
    for category in sorted({c["category"] for c in cases}):
        pairs = [
            (c, p)
            for c, p in zip(cases, predictions)
            if c["category"] == category
        ]
        groups[category] = metrics(
            [c for c, _ in pairs], [p for _, p in pairs]
        )
    result["by_category"] = groups
    result["errors"] = [
        {"text": c["text"], "expected": c["expected"], "predicted": p}
        for c, p in zip(cases, predictions)
        if p != c["expected"]
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["development", "holdout", "all"],
        default="development",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    sets = {
        "legacy_original": load_test_cases(str(ROOT / "benchmark_data.json")),
        "legacy_reviewed": reviewed_legacy(),
    }
    for split in (
        ["development", "holdout"] if args.split == "all" else [args.split]
    ):
        sets["challenge_" + split] = challenge(split)
    detectors = {
        profile: EnsembleDetector(profile=profile)
        for profile in [
            "legacy",
            "english",
            "english_extended",
            "corruption",
            "spoofing",
        ]
    }
    detectors.update(
        {strategy.value: GarbleDetector(strategy) for strategy in Strategy}
    )
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "notes": (
            "Historical/development data and a small authored family"
            "-separated challenge set. These results are not product"
            "ion precision estimates. Thresholds were selected witho"
            "ut using the challenge holdout."
        ),
        "input_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "benchmark_data.json",
                "label_overrides.json",
                "english_challenge.json",
            ]
        },
        "datasets": {
            name: {
                key: evaluate(detector, cases)
                for key, detector in detectors.items()
            }
            for name, cases in sets.items()
        },
    }
    if not args.details:
        for rows in report["datasets"].values():
            for result in rows.values():
                result.pop("by_category")
                result.pop("errors")
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(content)
    for name, rows in report["datasets"].items():
        print(
            name,
            {
                key: {
                    k: v
                    for k, v in rows[key].items()
                    if k not in ("by_category", "errors")
                }
                for key in ["legacy", "english", "english_extended"]
            },
        )


if __name__ == "__main__":
    main()
