"""Checks for statistical policy, data grouping and baseline calculations."""

import math
import tempfile
import unittest
from pathlib import Path

from .data import block_starts, download, samples
from .detectors import FOLDS, CharacterModel, next_float, threshold
from .metrics import bootstrap_difference, summarize, wilson


class StudyTests(unittest.TestCase):
    def test_cutoff_keeps_ties_and_respects_budget(self) -> None:
        scores = [(0.5, True, "scored")] * 99 + [(0.9, True, "scored")]
        selected = threshold(scores)
        self.assertGreater(selected, 0.5)
        self.assertLess(selected, 0.9)
        self.assertEqual(sum(s >= selected for s, _, _ in scores), 1)

    def test_full_range_can_require_external_keep_all(self) -> None:
        self.assertGreater(threshold([(1.0, True, "scored")]), 1.0)
        self.assertEqual(threshold([(1.0, False, "insufficient_evidence")]), 0)

    def test_empty_calibration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            threshold([])

    def test_next_float_adjacent(self) -> None:
        self.assertEqual(next_float(1.0), 1.0 + 2**-52)
        self.assertGreater(next_float(0.0), 0.0)
        with self.assertRaises(ValueError):
            next_float(float("inf"))

    def test_bigram_smoothing_matches_hand_calculation(self) -> None:
        model = CharacterModel(2, ["aa"])
        score, applicable, _ = model.score("aa")
        self.assertTrue(applicable)
        self.assertAlmostEqual(score, -math.log(11 / 271))
        unknown, _, _ = model.score("zz")
        self.assertAlmostEqual(unknown, math.log(27))

    def test_no_cross_document_transition(self) -> None:
        model = CharacterModel(2, ["ab", "cd"])
        self.assertEqual(model.counts["b"]["c"], 0)
        self.assertFalse(model.score("123")[1])

    def test_trigram_normalization(self) -> None:
        model = CharacterModel(3, ["abc"])
        self.assertAlmostEqual(model.score("ABC!")[0], -math.log(2 / 28))

    def test_blocks_nonoverlapping_and_spread(self) -> None:
        starts = block_starts(100000, 400)
        self.assertEqual(len(starts), 100)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 99600)
        self.assertTrue(all(b - a >= 400 for a, b in zip(starts, starts[1:])))
        self.assertEqual(block_starts(399, 400), [])

    def test_short_documents_not_padded(self) -> None:
        rows = samples(
            [
                {
                    "text": "a" * 500,
                    "label": 1,
                    "doc_id": "x",
                    "family": "gibberish",
                    "domain": "human_produced",
                }
            ]
        )
        self.assertEqual({r["view"] for r in rows}, {0, 100, 400})
        self.assertTrue(all(r["start"] == 0 for r in rows))

    def test_fold_roles_disjoint(self) -> None:
        self.assertEqual(
            {test for test, _, _ in FOLDS}, {"bible", "secreta", "wiki"}
        )
        for fold in FOLDS:
            self.assertEqual(len(set(fold)), 3)

    def test_intervals_and_paired_resampling(self) -> None:
        lower, upper = wilson(38, 38)
        self.assertLess(lower, 0.95)
        self.assertAlmostEqual(upper, 1.0)
        self.assertEqual(bootstrap_difference([1, 0], [1, 0]), [0.0, 0.0])
        self.assertEqual(bootstrap_difference([1, 1], [0, 0]), [1.0, 1.0])

    def test_macro_fpr_uses_documents_not_passage_pool(self) -> None:
        rows = []
        for doc, label, decisions in [
            ("p", 1, [1]),
            ("a", 0, [1]),
            ("b", 0, [0] * 9),
        ]:
            for i, value in enumerate(decisions):
                rows.append(
                    {
                        "id": str(i) + doc,
                        "doc_id": doc,
                        "view": 400,
                        "experiment": "default",
                        "method": "english",
                        "fold": "all",
                        "label": label,
                        "predicted": value,
                        "applicable": 1,
                    }
                )
        result = summarize(rows)[0]
        self.assertEqual(result["macro_document_fpr"], 0.5)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["negative_blocks"], 10)

    def test_corrupted_cache_rejected_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "README.md").write_text("corrupted cache")
            with self.assertRaisesRegex(ValueError, "Cached checksum"):
                download(root)


if __name__ == "__main__":
    unittest.main()
