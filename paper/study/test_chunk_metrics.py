"""Check descriptive metrics against independent saved confusion counts."""

import json
import unittest

from .chunk_metrics import from_counts, summarize
from .data import ROOT
from .full_corpus import METHODS, read_parts


class ChunkMetricTests(unittest.TestCase):
    def test_always_keep_exposes_accuracy_imbalance(self) -> None:
        result = from_counts(0, 173, 0, 5200)
        self.assertAlmostEqual(result["accuracy"], 5200 / 5373)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)
        self.assertIsNone(result["precision"])

    def test_asymmetric_confusion_matrix(self) -> None:
        result = from_counts(3, 1, 2, 4)
        self.assertEqual(result["accuracy"], 0.7)
        self.assertEqual(result["precision"], 0.6)
        self.assertEqual(result["recall"], 0.75)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], (0.75 + 4 / 6) / 2)

    def test_invalid_counts(self) -> None:
        for counts in [(0, 0, 0, 0), (-1, 2, 0, 0), (1.5, 0, 1, 1)]:
            with self.assertRaises(ValueError):
                from_counts(*counts)

    def test_confusions_match_every_saved_target_prediction(self) -> None:
        output = ROOT / "full-results"
        progress = json.loads((output / "progress.json").read_text())
        docs = json.loads((output / "documents.json").read_text())
        counts = {m: [0, 0, 0, 0] for m in METHODS}
        for doc in docs:
            if doc["scope"] == "other_language":
                continue
            for row in read_parts(
                output, progress["documents"][doc["doc_id"]]["parts"]
            ):
                for method in METHODS:
                    predicted = (
                        row["hf"][method]
                        if method.startswith("hf_")
                        else row["pygarble"][method]["flag"]
                    )
                    index = (
                        (0 if predicted else 1)
                        if doc["label"]
                        else (2 if predicted else 3)
                    )
                    counts[method][index] += 1
        summary = json.loads((output / "summary.json").read_text())
        for row in summarize(summary):
            if row["method"] == "keep_all":
                continue
            self.assertEqual(
                [row[k] for k in ("tp", "fn", "fp", "tn")],
                counts[row["method"]],
            )


if __name__ == "__main__":
    unittest.main()
