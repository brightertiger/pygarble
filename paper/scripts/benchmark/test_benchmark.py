"""One behaviour check per benchmark script; runs without the raw data."""

import json
import unittest

import numpy as np

from . import data, footprint, metrics, rules, verify
from .methods import specs


class DataTests(unittest.TestCase):
    def test_manifest_lists_all_of_b_in_file_order(self) -> None:
        manifest = json.loads(data.MANIFEST.read_text())
        ids = manifest["B_ids"]
        self.assertEqual(ids, list(manifest["B_labels"]))
        self.assertEqual(ids, sorted(ids, key=data._order))
        counts = manifest["counts"]
        self.assertEqual((counts["A"], counts["B"]), (3513, 9971))
        self.assertEqual(sum(manifest["B_labels"].values()), 4000)


class MethodTests(unittest.TestCase):
    def test_rules_and_pygarble_wrapper(self) -> None:
        tweet = "@user that was the best game of the season, honestly"
        self.assertIn("word_count", rules.gopher_failures(tweet))
        prose = " ".join(["The cat sat on the warm mat by the door."] * 5)
        self.assertEqual(rules.c4_failures(prose, frozenset()), [])
        self.assertIn(
            "curly_bracket", rules.c4_failures(prose + " {", frozenset())
        )
        method = specs()["english_fusion"].build()
        scores, flags = method.run([tweet, "qwkjzx vbnmrt plokij"])
        self.assertEqual(flags, [False, True])
        self.assertGreater(scores[1], scores[0])


class RunTests(unittest.TestCase):
    def test_metrics_at_the_default_decision(self) -> None:
        ids = [f"train:{i}" for i in range(100)]
        labels = {i: int(n % 2) for n, i in enumerate(ids)}
        manifest = {"A_ids": ["a:0", "a:1"], "B_ids": ids, "B_labels": labels}
        flags = np.array([labels[i] == 1 for i in ids])
        flags[0] = True  # one false positive among 50 tweets
        predictions = {
            "english_fusion": {
                "A": (np.zeros(2), np.array([True, False])),
                "B": (flags.astype(float), flags),
            }
        }
        result = metrics.compute(
            manifest, predictions, {"english_fusion": "profile"}
        )
        method = result["methods"]["english_fusion"]
        self.assertEqual(method["A_capture"]["value"], 0.5)
        self.assertEqual(method["B_recall"]["value"], 1.0)
        self.assertEqual(method["B_fpr"]["value"], 0.02)
        self.assertEqual(method["B_precision"]["value"], round(50 / 51, 6))
        low, high = method["B_fpr"]["ci"]
        self.assertLessEqual(low, 0.02)
        self.assertGreaterEqual(high, 0.02)


class VerifyTests(unittest.TestCase):
    def test_differences_reports_a_changed_number(self) -> None:
        saved = {"m": {"auroc": {"value": 0.9, "ci": [0.8, 1.0]}}}
        fresh = {"m": {"auroc": {"value": 0.9, "ci": [0.8, 0.99]}}}
        self.assertEqual(verify.differences(saved, saved), [])
        self.assertEqual(
            verify.differences(saved, fresh), ["/m/auroc/ci[1]: 1.0 != 0.99"]
        )


class FootprintTests(unittest.TestCase):
    def test_timing_and_install_size(self) -> None:
        method = specs()["english"].build()
        ms, runs = footprint.time_method(method, ["hello there"] * 20, runs=2)
        self.assertGreater(ms, 0)
        self.assertEqual(len(runs), 2)
        size = footprint.install_bytes("english", ("pygarble",), data.OUTPUT)
        self.assertGreater(size, 100_000)


if __name__ == "__main__":
    unittest.main()
