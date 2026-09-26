"""Post hoc descriptive chunk metrics; no new inference or threshold tuning."""

import json

from .data import ROOT, write_json


def from_counts(tp: int, fn: int, fp: int, tn: int) -> dict:
    counts = (tp, fn, fp, tn)
    if any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError("Expected nonnegative integer counts with data")
    positive = tp + fn
    negative = tn + fp
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": (tp + tn) / sum(counts),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
    }


def summarize(summary: dict) -> list:
    rows = []
    for row in summary["primary"]:
        tp = row["positive_flagged_chunks"]
        fp = row["english_false_flag_chunks"]
        rows.append(
            {
                "method": row["method"],
                **from_counts(
                    tp,
                    row["positive_chunks"] - tp,
                    fp,
                    row["english_chunks"] - fp,
                ),
            }
        )
    reference = summary["primary"][0]
    rows.append(
        {
            "method": "keep_all",
            **from_counts(
                0, reference["positive_chunks"], 0, reference["english_chunks"]
            ),
        }
    )
    return rows


def main() -> None:
    output = ROOT / "full-results"
    summary = json.loads((output / "summary.json").read_text())
    write_json(
        output / "chunk-metrics.json",
        {
            "unit": "dependent chunks of up to 400 characters",
            "scope": "released gibberish plus four English control files",
            "analysis": "post hoc descriptive reporting; no new predictions",
            "rows": summarize(summary),
        },
    )


if __name__ == "__main__":
    main()
