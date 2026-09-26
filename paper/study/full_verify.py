"""Verify all full-corpus artifacts and replay a fixed offline subset."""

import argparse
import gzip
import json
import math
from pathlib import Path

from .data import ROOT, download, write_json
from .detectors import make_model
from .full_corpus import (
    METHODS,
    PACKAGE_METHODS,
    chunks,
    fingerprint,
    load_documents,
    read_parts,
)
from .full_metrics import document_summary, summarize_documents
from .hf_backend import HFModel, policies, verify_model


def verify(output: Path, replay: bool) -> dict:
    download(ROOT / ".cache")
    verify_model(ROOT / ".cache/hf-model")
    assert json.loads((output / "code-manifest.json").read_text()) == (
        fingerprint()
    )
    docs = load_documents(ROOT / ".cache")
    manifest = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    assert json.loads((output / "documents.json").read_text()) == manifest
    progress = json.loads((output / "progress.json").read_text())
    assert progress["complete"]
    assert set(progress["documents"]) == {d["doc_id"] for d in docs}
    total = 0
    doc_rows = []
    replay_rows = []
    for doc in docs:
        rows = read_parts(
            output, progress["documents"][doc["doc_id"]]["parts"]
        )
        expected = chunks(doc)
        assert len(rows) == len(expected)
        for source, row in zip(expected, rows):
            assert {k: row[k] for k in source if k != "text"} == {
                k: v for k, v in source.items() if k != "text"
            }
            hf = row["hf"]
            assert 2 <= hf["tokens"] <= 512
            for key, value in policies(hf["probabilities"]).items():
                assert hf[key] == value
            assert set(row["pygarble"]) == set(PACKAGE_METHODS)
            for model in row["pygarble"].values():
                assert math.isfinite(model["score"])
                assert model["flag"] == int(
                    model["applicable"] and model["score"] >= 0.5
                )
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            replay_rows.append((expected[index], rows[index]))
        doc_rows.extend(document_summary(doc, rows, m) for m in METHODS)
        total += len(rows)
    stored_docs = json.loads(
        gzip.decompress((output / "document-results.json.gz").read_bytes())
    )
    assert stored_docs == doc_rows
    assert json.loads((output / "summary.json").read_text()) == (
        summarize_documents(doc_rows)
    )
    assert total == progress["completed_chunks"] == 79969
    result = {
        "documents": len(docs),
        "chunks_verified": total,
        "pygarble_decisions_verified": total * len(PACKAGE_METHODS),
        "hf_policy_decisions_verified": total * 2,
        "source_and_model_hashes_verified": True,
        "coverage_and_summaries_recomputed": True,
        "replay_performed": replay,
    }
    if replay:
        hf_model = HFModel(ROOT / ".cache/hf-model")
        models = {m: make_model(m, []) for m in PACKAGE_METHODS}
        max_delta = 0.0
        for start in range(0, len(replay_rows), 8):
            batch = replay_rows[start : start + 8]
            new = hf_model.predict_batch([s["text"] for s, _ in batch])
            for (source, saved), predicted in zip(batch, new):
                assert predicted["tokens"] == saved["hf"]["tokens"]
                for policy in ("hf_all", "hf_strict", "label"):
                    assert predicted[policy] == saved["hf"][policy]
                delta = max(
                    abs(a - b)
                    for a, b in zip(
                        predicted["probabilities"],
                        saved["hf"]["probabilities"],
                    )
                )
                assert delta <= 1e-5
                max_delta = max(max_delta, delta)
                for name, model in models.items():
                    score, applicable, status = model.score(source["text"])
                    old = saved["pygarble"][name]
                    assert (score, applicable, status) == (
                        old["score"],
                        old["applicable"],
                        old["status"],
                    )
                    assert int(model.detector.predict(source["text"])) == (
                        old["flag"]
                    )
        result.update(
            replay_inputs=len(replay_rows),
            replay_max_probability_difference=max_delta,
            replay_all_decisions_identical=True,
            replay_public_api_decisions=len(replay_rows) * len(models),
        )
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "full-results")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.replay), indent=2))


if __name__ == "__main__":
    main()
