"""Uncapped published corpus: all labelled documents, identical chunks."""

import argparse
import gzip
import importlib.metadata
import json
import platform
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import ROOT, digest, download, write_json
from .detectors import make_model
from .hf_backend import HFModel, verify_model

PACKAGE_METHODS = (
    "english",
    "english_extended",
    "legacy",
    "word_lookup",
    "entropy_based",
)
METHODS = PACKAGE_METHODS + ("hf_all", "hf_strict")


def load_documents(cache: Path) -> list:
    docs = []
    for archive, label in [
        ("gibberish_transcriptions.zip", 1),
        ("meaningful.zip", 0),
    ]:
        with zipfile.ZipFile(cache / archive) as z:
            for name in sorted(z.namelist()):
                selected = (
                    bool(re.fullmatch(r"Gibberish - D[AC]_\d+\.txt", name))
                    if label
                    else name.startswith("texts/") and name.endswith(".txt")
                )
                if not selected:
                    continue
                raw = z.read(name)
                text = " ".join(raw.decode("utf-8-sig").split())
                if not text:
                    raise ValueError("Empty published text: " + name)
                parts = Path(name).stem.split(" - ")
                doc_id = (
                    "gib-" + parts[-1]
                    if label
                    else "text-" + digest(name.encode())[:12]
                )
                docs.append(
                    {
                        "doc_id": doc_id,
                        "archive": archive,
                        "member": name,
                        "label": label,
                        "language": "invented" if label else parts[1],
                        "era": "experiment" if label else parts[0],
                        "genre": "invented" if label else parts[2],
                        "scope": (
                            "positive"
                            if label
                            else (
                                "english"
                                if parts[1] == "English"
                                else "other_language"
                            )
                        ),
                        "raw_sha256": digest(raw),
                        "text": text,
                        "normalized_sha256": digest(text.encode()),
                        "characters": len(text),
                        "chunk_count": (len(text) + 399) // 400,
                    }
                )
    if Counter(d["label"] for d in docs) != {0: 71, 1: 38}:
        raise ValueError("Corpus membership changed")
    # Finish the target-language analysis first, without sampling either group.
    return sorted(
        docs,
        key=lambda d: (
            {"positive": 0, "english": 1, "other_language": 2}[d["scope"]],
            d["doc_id"],
        ),
    )


def chunks(doc: dict) -> list:
    text = doc["text"]
    rows = []
    for start in range(0, len(text), 400):
        part = text[start : start + 400]
        rows.append(
            {
                "id": "{}:{}".format(doc["doc_id"], start),
                "start": start,
                "stop": start + len(part),
                "characters": len(part),
                "text_sha256": digest(part.encode()),
                "text": part,
            }
        )
    if "".join(row["text"] for row in rows) != text:
        raise AssertionError("Incomplete text coverage")
    return rows


def fingerprint() -> dict:
    paths = [
        ROOT / name
        for name in (
            "full_corpus.py",
            "full_metrics.py",
            "hf_backend.py",
            "full_corpus_protocol.md",
            "hf_model.json",
            "sources.json",
            "data.py",
            "detectors.py",
            "metrics.py",
        )
    ]
    paths += sorted((ROOT.parents[1] / "pygarble").rglob("*.py"))
    return {
        str(p.relative_to(ROOT.parents[1])): digest(p.read_bytes())
        for p in paths
    }


def save_parts(output: Path, doc_id: str, predictions: list) -> list:
    names = []
    for index, start in enumerate(range(0, len(predictions), 1000)):
        name = "{}.{}.jsonl.gz".format(doc_id, index)
        raw = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            for row in predictions[start : start + 1000]
        )
        compressed = gzip.compress(raw, mtime=0)
        if len(compressed) > 480000:
            raise ValueError("Prediction shard exceeds repository size limit")
        path = output / "predictions" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(compressed)
        names.append(
            {"file": "predictions/" + name, "sha256": digest(compressed)}
        )
    return names


def read_parts(output: Path, parts: list) -> list:
    rows = []
    for part in parts:
        raw = (output / part["file"]).read_bytes()
        if digest(raw) != part["sha256"]:
            raise ValueError("Prediction checksum mismatch")
        rows.extend(
            json.loads(line) for line in gzip.decompress(raw).splitlines()
        )
    return rows


def evaluate(args: argparse.Namespace) -> None:
    from .full_metrics import analyze

    download(args.cache)
    verify_model(args.model_dir)
    docs = load_documents(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = fingerprint()
    manifest_path = args.output / "code-manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != frozen
    ):
        raise ValueError("Evaluation code changed; use a new output directory")
    write_json(manifest_path, frozen)
    manifest = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    write_json(args.output / "documents.json", manifest)
    progress_path = args.output / "progress.json"
    completed = (
        json.loads(progress_path.read_text())["documents"]
        if progress_path.exists()
        else {}
    )
    models = {name: make_model(name, []) for name in PACKAGE_METHODS}
    hf = HFModel(args.model_dir)
    overall_start = time.monotonic()
    chunk_counter = 0
    for doc in docs:
        doc_id = doc["doc_id"]
        selected = chunks(doc)
        if doc_id in completed:
            stored = read_parts(args.output, completed[doc_id]["parts"])
            if len(stored) != len(selected) or any(
                a["text_sha256"] != b["text_sha256"]
                for a, b in zip(stored, selected)
            ):
                raise ValueError("Resume corpus mismatch: " + doc_id)
            chunk_counter += len(selected)
            continue
        started = time.monotonic()
        predictions = []
        for start in range(0, len(selected), 8):
            batch = selected[start : start + 8]
            neural = hf.predict_batch([row["text"] for row in batch])
            for row, nn in zip(batch, neural):
                score_rows = {}
                for name, model in models.items():
                    score, applicable, status = model.score(row["text"])
                    score_rows[name] = {
                        "score": score,
                        "applicable": applicable,
                        "status": status,
                        "flag": int(applicable and score >= 0.5),
                    }
                predictions.append(
                    {
                        **{k: v for k, v in row.items() if k != "text"},
                        "hf": nn,
                        "pygarble": score_rows,
                    }
                )
            if start and start % 400 == 0:
                print(
                    "Progress {}: {}/{} chunks".format(
                        doc_id, min(start + 8, len(selected)), len(selected)
                    ),
                    flush=True,
                )
        parts = save_parts(args.output, doc_id, predictions)
        completed[doc_id] = {
            "chunks": len(predictions),
            "parts": parts,
            "elapsed_seconds": time.monotonic() - started,
        }
        chunk_counter += len(predictions)
        write_json(
            progress_path,
            {
                "complete": False,
                "documents": completed,
                "completed_chunks": chunk_counter,
            },
        )
        print(
            "Completed {}/{} documents; {} chunks; {} ({:.1f}s)".format(
                len(completed),
                len(docs),
                chunk_counter,
                doc["member"],
                time.monotonic() - started,
            ),
            flush=True,
        )
        if len(completed) == 42:
            print(
                "All English controls and gibberish documents complete.",
                flush=True,
            )
    write_json(
        progress_path,
        {
            "complete": True,
            "documents": completed,
            "completed_chunks": chunk_counter,
        },
    )
    write_json(
        args.output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": 4,
            "batch_size": 8,
            "device": "cpu",
            "attention": "sdpa",
            "dtype": "float32",
            "model": json.loads((ROOT / "hf_model.json").read_text())[
                "revision"
            ],
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                    "tokenizers",
                ]
            },
            "elapsed_seconds_this_invocation": time.monotonic()
            - overall_start,
            "utc": datetime.now(timezone.utc).isoformat(),
            "new_human_label_audit": False,
        },
    )
    analyze(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / ".cache/hf-model"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "full-results")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.download_model:
        verify_model(args.model_dir, allow_download=True)
    elif args.analyze_only:
        from .full_metrics import analyze

        analyze(args.output)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
