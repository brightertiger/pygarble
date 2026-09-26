"""Descriptive estimates with source dependence kept explicit."""

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def wilson(success: int, count: int) -> List[float]:
    if count == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = success / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2))
    half /= denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_difference(a: List[int], b: List[int]) -> List[float]:
    if len(a) != len(b) or not a:
        raise ValueError("Paired nonempty outcomes required")
    rng = random.Random(20260926)
    delta = [x - y for x, y in zip(a, b)]
    values = sorted(
        sum(rng.choice(delta) for _ in delta) / len(delta) for _ in range(2000)
    )
    return [values[49], values[1949]]


def summarize(predictions: list) -> list:
    grouped: Dict[Tuple, list] = defaultdict(list)
    for row in predictions:
        grouped[
            (row["experiment"], row["method"], row["view"], row["fold"])
        ].append(row)
    reference = {
        (row["view"], row["id"]): row["predicted"]
        for row in predictions
        if row["experiment"] == "default"
        and row["method"] == "english"
        and row["label"]
    }
    result = []
    for (experiment, method, view, fold), rows in sorted(grouped.items()):
        positives = sorted(
            [r for r in rows if r["label"]], key=lambda r: r["id"]
        )
        negatives = [r for r in rows if not r["label"]]
        by_source: Dict[str, list] = defaultdict(list)
        for row in negatives:
            by_source[row["doc_id"]].append(row)
        source_rates = {
            source: {
                "n": len(values),
                "false_positives": sum(r["predicted"] for r in values),
                "fpr": sum(r["predicted"] for r in values) / len(values),
            }
            for source, values in sorted(by_source.items())
        }
        tp = sum(r["predicted"] for r in positives)
        n = len(positives)
        result.append(
            {
                "experiment": experiment,
                "method": method,
                "view": view,
                "fold": fold,
                "positive_documents": n,
                "true_positives": tp,
                "recall": tp / n if n else None,
                "recall_wilson95_conditional": wilson(tp, n),
                "paired_recall_delta_vs_default_english_ci95": (
                    bootstrap_difference(
                        [r["predicted"] for r in positives],
                        [reference[(view, r["id"])] for r in positives],
                    )
                    if n
                    else None
                ),
                "negative_documents": len(by_source),
                "negative_blocks": len(negatives),
                "false_positives": sum(r["predicted"] for r in negatives),
                "macro_document_fpr": (
                    sum(v["fpr"] for v in source_rates.values())
                    / len(source_rates)
                    if source_rates
                    else None
                ),
                "by_negative_document": source_rates,
                "coverage": sum(r["applicable"] for r in rows) / len(rows),
            }
        )
    return result
