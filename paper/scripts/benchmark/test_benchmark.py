"""One behaviour check per benchmark script; runs without the raw data."""

import json
import unittest

import numpy as np

from . import data, footprint, metrics, tune, verify
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
    def test_pygarble_wrappers_and_tuned_members(self) -> None:
        every = specs()
        self.assertEqual(
            list(every),
            [
                "english",
                "english_fusion",
                "gibberish_detector",
                "lingua",
                "english_tuned",
            ],
        )
        tuned = every["english_tuned"].build().detector
        english = every["english"].build().detector
        self.assertEqual(
            [s.value for s in tuned.strategies],
            [
                s.value
                for s in english.strategies
                if s.value not in ("mojibake", "control_characters")
            ],
        )
        self.assertEqual((tuned.voting, tuned.threshold), ("any", 0.5))
        tweet = "@user that was the best game of the season, honestly"
        method = every["english_fusion"].build()
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


class TuneTests(unittest.TestCase):
    def test_cached_weighted_vote_matches_ensemble_detector(self) -> None:
        texts = [
            "@user that was the best game of the season, honestly",
            "qwkjzx vbnmrt plokij",
            "asdf asdf asdf",
            "Ã¢â‚¬Å“helloÃ¢â‚¬",
            "!!!",
            "the cat sat on the mat",
            "x",
            "Please call me back tomorrow about the invoice.",
        ]
        rows = [data.Row(f"t:{i}", t, 0) for i, t in enumerate(texts)]
        names = tune.strategy_names()
        records = tune.score_rows(rows, names)
        scores = np.array([r[2] for r in records]).reshape(len(names), -1).T
        flags = np.array([r[3] for r in records]).reshape(len(names), -1).T
        self.assertFalse(flags.all())  # some members are inapplicable
        rng = np.random.default_rng(1)
        for _ in range(5):
            weights = rng.choice(np.arange(0, 1.05, 0.05).round(2), 32)
            params = {
                "weights": dict(zip(names, map(float, weights))),
                "threshold": 0.4,
            }
            got, decided = tune.weighted((scores, flags), weights, 0.4)
            detector = tune.ensemble(params)
            self.assertEqual(list(got), detector.predict_proba(texts))
            self.assertEqual(list(decided), detector.predict(texts))


if __name__ == "__main__":
    unittest.main()
