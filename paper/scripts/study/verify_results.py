"""Verify saved metrics, frozen sources and public API decisions."""

import csv
import gzip
import io
import json
from pathlib import Path

from .data import ROOT, digest, documents, download, samples, write_json
from .detectors import DEFAULTS, make_model
from .metrics import summarize


def load_predictions(path: Path) -> list:
    stream = io.StringIO(gzip.decompress(path.read_bytes()).decode())
    rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ("label", "view", "applicable", "predicted"):
            row[name] = int(row[name])
        for name in ("score", "threshold"):
            row[name] = float(row[name])
    return rows


def main() -> None:
    output = ROOT / "results"
    frozen = json.loads((output / "code-manifest.json").read_text())
    for name, expected in frozen.items():
        source = ROOT.parents[1] / name
        if not source.is_file() or digest(source.read_bytes()) != expected:
            raise ValueError(
                "Frozen source differs: {}. Use the historical checkout "
                "documented in paper/study/README.md.".format(name)
            )
    predictions = load_predictions(output / "predictions.csv.gz")
    expected_summary = json.loads((output / "summary.json").read_text())
    assert summarize(predictions) == expected_summary
    download(ROOT / ".cache")
    docs = documents(ROOT / ".cache")
    rows = {r["id"]: r for r in samples(docs)}
    models = {
        name: make_model(name, []) for name in DEFAULTS if name != "keep_all"
    }
    api_checks = 0
    for row in predictions:
        if row["experiment"] != "default" or row["method"] == "keep_all":
            continue
        model = models[row["method"]]
        decision = model.detector.predict(rows[row["id"]]["text"])
        assert decision == bool(row["predicted"]), row["id"]
        api_checks += 1
    reproduced = []
    for name in [
        "predictions.csv.gz",
        "summary.json",
        "calibration.json",
        "documents.json",
        "records.json",
        "overlap.json",
        "report.md",
    ]:
        other = ROOT / "reproduction" / name
        if not other.exists():
            raise FileNotFoundError("Run the documented reproduction first")
        assert (output / name).read_bytes() == other.read_bytes(), name
        reproduced.append(name)
    write_json(
        ROOT / "validation.json",
        {
            "frozen_source_files_verified": len(frozen),
            "prediction_rows": len(predictions),
            "public_predict_api_decisions_checked": api_checks,
            "saved_summary_recomputed": True,
            "byte_identical_reproduction_artifacts": reproduced,
            "new_human_label_audit": False,
        },
    )
    print("Saved results, frozen code and public API decisions verified.")


if __name__ == "__main__":
    main()
