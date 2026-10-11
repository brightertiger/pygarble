"""One behaviour check per benchmark script; runs without the raw data."""

import json
import unittest

from . import data


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


if __name__ == "__main__":
    unittest.main()
