"""Tests for full coverage, aggregation and the neural label policies."""

import unittest
from unittest.mock import patch

from .data import digest
from .full_corpus import METHODS, chunks, fingerprint
from .full_metrics import document_summary, majority, summarize_documents
from .full_verify import verify
from .hf_backend import policies
from .paths import REPOSITORY, ROOT, SCRIPTS
from .run import code_manifest


class FullCorpusTests(unittest.TestCase):
    def test_manifests_cover_relocated_code_and_original_assets(self) -> None:
        full = fingerprint()
        preliminary = code_manifest()
        for manifest in (full, preliminary):
            for name in ("data.py", "paths.py", "detectors.py"):
                self.assertEqual(
                    manifest["paper/scripts/study/" + name],
                    digest((SCRIPTS / name).read_bytes()),
                )
            self.assertEqual(
                manifest["paper/study/sources.json"],
                digest((ROOT / "sources.json").read_bytes()),
            )
            self.assertIn("pygarble/__init__.py", manifest)
            self.assertTrue(all((REPOSITORY / p).is_file() for p in manifest))
            self.assertFalse(
                any(
                    p.startswith("paper/study/") and p.endswith(".py")
                    for p in manifest
                )
            )

    def test_frozen_manifest_mismatch_fails_before_downloading(self) -> None:
        with patch("paper.scripts.study.full_verify.download") as download:
            with self.assertRaisesRegex(ValueError, "historical checkout"):
                verify(ROOT / "full-results", replay=False)
            download.assert_not_called()

    def test_tail_and_complete_reconstruction(self) -> None:
        text = "abc " * 203
        rows = chunks({"doc_id": "x", "text": text})
        self.assertEqual([r["characters"] for r in rows], [400, 400, 12])
        self.assertEqual("".join(r["text"] for r in rows), text)
        self.assertEqual(rows[-1]["stop"], len(text))

    def test_exact_multiple_does_not_add_empty_tail(self) -> None:
        self.assertEqual(len(chunks({"doc_id": "x", "text": "a" * 800})), 2)

    def test_majority_rule_and_ties(self) -> None:
        self.assertEqual(majority([0, 1]), 1)
        self.assertEqual(majority([0, 0, 1]), 0)
        with self.assertRaises(ValueError):
            majority([])

    def test_label_mapping_is_explicit(self) -> None:
        mild = policies([0.1, 0.7, 0.1, 0.1])
        self.assertEqual((mild["hf_all"], mild["hf_strict"]), (1, 0))
        clean = policies([0.7, 0.1, 0.1, 0.1])
        self.assertEqual((clean["hf_all"], clean["hf_strict"]), (0, 0))
        for values in ([0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]):
            self.assertEqual(policies(values)["hf_strict"], 1)

    def test_native_argmax_is_not_sum_threshold(self) -> None:
        result = policies([0.4, 0.3, 0.2, 0.1])
        self.assertGreater(result["score"], 0.5)
        self.assertEqual(result["hf_all"], 0)

    def test_invalid_probabilities_fail(self) -> None:
        for values in ([0.5, 0.5], [0.2] * 4, [float("nan"), 0, 0, 0]):
            with self.assertRaises(ValueError):
                policies(values)

    def test_tail_excluded_rate_is_separate(self) -> None:
        doc = dict(
            doc_id="x",
            member="x",
            label=1,
            scope="positive",
            language="invented",
            era="experiment",
            genre="invented",
        )
        rows = [
            {"characters": 400, "hf": {"hf_all": 0}},
            {"characters": 10, "hf": {"hf_all": 1}},
        ]
        result = document_summary(doc, rows, "hf_all")
        self.assertEqual(result["majority_decision"], 1)
        self.assertEqual(result["full_width_majority"], 0)
        self.assertEqual(result["flagged_fraction"], 0.5)
        self.assertEqual(result["full_width_fraction"], 0.0)

    def test_macro_rates_do_not_weight_long_sources_more(self) -> None:
        rows = []
        for method in METHODS:
            for name, scope, chunks_n, flagged in [
                ("gib", "positive", 3, 3),
                ("long", "english", 4000, 0),
                ("short", "english", 1, 1),
                ("foreign", "other_language", 9, 9),
            ]:
                rows.append(
                    dict(
                        doc_id=name,
                        method=method,
                        label=int(scope == "positive"),
                        scope=scope,
                        language="English" if scope == "english" else name,
                        chunks=chunks_n,
                        flagged=flagged,
                        flagged_fraction=flagged / chunks_n,
                        majority_decision=int(flagged * 2 >= chunks_n),
                        any_decision=int(flagged > 0),
                        full_width_fraction=flagged / chunks_n,
                    )
                )
        summary = summarize_documents(rows)["primary"][0]
        self.assertEqual(summary["english_macro_chunk_fpr"], 0.5)
        self.assertEqual(summary["english_false_flag_chunks"], 1)
        self.assertEqual(summary["english_chunks"], 4001)
        self.assertEqual(summary["other_language_macro_fpr"], 1.0)


if __name__ == "__main__":
    unittest.main()
