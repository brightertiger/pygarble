"""One behaviour check per benchmark script; runs without the raw data."""

import json
import unittest

import numpy as np

from . import calibrate, data, footprint, metrics, rules, verify
from .methods import specs


class DataTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_matches_manifest(
        self,
    ) -> None:
        rows = [
            data.Row(f"train:{i}", "x", int(i % 5 < 2)) for i in range(100)
        ]
        dev, test = data.split_b(rows)
        self.assertEqual((dev, test), data.split_b(rows))
        self.assertNotEqual(dev, data.split_b(rows, seed=1)[0])
        self.assertEqual(len(dev), 30)
        self.assertEqual(sum(int(i.split(":")[1]) % 5 < 2 for i in dev), 12)
        self.assertFalse(set(dev) & set(test))
        manifest = json.loads(data.MANIFEST.read_text())
        labels = manifest["B_labels"]
        saved = [data.Row(i, "", labels[i]) for i in labels]
        self.assertEqual(
            data.split_b(saved),
            (manifest["B_dev_ids"], manifest["B_test_ids"]),
        )


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
    def test_metrics_on_a_perfect_separator(self) -> None:
        dev = [f"train:{i}" for i in range(200)]
        test = [f"test:{i}" for i in range(100)]
        labels = {i: int(n % 2) for n, i in enumerate(dev + test)}
        manifest = {
            "A_ids": ["a:0", "a:1"],
            "B_dev_ids": dev,
            "B_test_ids": test,
            "B_labels": labels,
        }

        def part(ids: list) -> tuple:
            y = np.array([labels.get(i, 1) for i in ids])
            return y + 0.001 * np.arange(len(ids)), y == 1

        predictions = {
            "english_fusion": {
                "A": part(manifest["A_ids"]),
                "B_dev": part(dev),
                "B_test": part(test),
            }
        }
        result, curves = metrics.compute(
            manifest, predictions, {"english_fusion": "profile"}
        )
        method = result["methods"]["english_fusion"]
        self.assertEqual(method["auroc"]["value"], 1.0)
        self.assertEqual(method["points"]["fpr1"]["dev_fpr"], 0.01)
        self.assertTrue(method["points"]["fpr1"]["reachable"])
        point = method["points"]["fpr0.1"]
        self.assertEqual(point["B_precision"]["value"], 1.0)
        self.assertEqual(point["A_capture"]["value"], 1.0)
        self.assertTrue(result["checkpoint"]["passed"])
        self.assertEqual(curves["english_fusion"][-1][2], 1.0)


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


class CalibrateTests(unittest.TestCase):
    def test_threshold_meets_target_on_calibration_sample(self) -> None:
        sample = {
            "clean": ["the meeting moved to friday afternoon"] * 50,
            "garbled": ["qwkjzx vbnmrt plokij"] * 50,
        }
        result = calibrate.calibrate_profile("english_fusion", sample)
        self.assertTrue(result["constraint_met"])
        self.assertEqual(result["dev_recall"], 1.0)
        detector = result["detector"]
        self.assertEqual(detector.predict(sample["garbled"][:1]), [True])


if __name__ == "__main__":
    unittest.main()
