"""Build review-ready Markdown, TeX and PDF from measured result tables."""

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {
    "keep_all": "Always keep",
    "english": "English",
    "english_extended": "Extended",
    "legacy": "Legacy",
    "word_lookup": "Word lookup",
    "entropy_based": "Entropy",
    "char_bigram": "Character bigram",
    "char_trigram": "Character trigram",
}


def table(header: list, rows: list) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render() -> str:
    summary = json.loads((ROOT / "results/summary.json").read_text())
    runtime = json.loads((ROOT / "results/runtime.json").read_text())
    defaults = {
        (r["method"], r["view"]): r
        for r in summary
        if r["experiment"] == "default"
    }
    methods = list(NAMES)[:6]
    rows = []
    for method in methods:
        row = defaults[(method, 400)]
        rows.append(
            [
                NAMES[method],
                "{}/{}".format(
                    row["true_positives"], row["positive_documents"]
                ),
                "{:.1%}".format(row["recall"]),
                *[
                    "{:.0%}".format(row["by_negative_document"][d]["fpr"])
                    for d in ("kjv", "net", "secreta", "wiki")
                ],
            ]
        )
    default_table = table(
        [
            "Method",
            "Detected",
            "Recall",
            "KJV FPR",
            "NET FPR",
            "Secreta FPR",
            "Wiki FPR",
        ],
        rows,
    )
    rows = []
    for method in methods:
        rows.append(
            [NAMES[method]]
            + [
                "{}/{}".format(
                    defaults[(method, view)]["true_positives"],
                    defaults[(method, view)]["positive_documents"],
                )
                for view in (100, 400, 800, 0)
            ]
        )
    sensitivity_table = table(
        ["Method", "100 chars", "400 chars", "800 chars", "Whole positive"],
        rows,
    )
    rows = []
    lookup = {
        (r["method"], r["fold"]): r
        for r in summary
        if r["experiment"] == "calibrated"
    }
    for method in list(NAMES)[1:]:
        values = [NAMES[method]]
        for fold in ("bible", "secreta", "wiki"):
            row = lookup[(method, fold)]
            values.append(
                "{:.1%} / {:.1%}".format(
                    row["recall"], row["macro_document_fpr"]
                )
            )
        rows.append(values)
    calibration_table = table(
        [
            "Method",
            "Bible recall / FPR",
            "Secreta recall / FPR",
            "Wiki recall / FPR",
        ],
        rows,
    )
    rows = [
        [
            NAMES[r["method"]],
            "{:.3f}".format(r["warm_median_ms"]),
            "{:.3f}".format(r["warm_p95_ms"]),
            "{:.1f}".format(r["peak_process_rss_bytes"] / 1024**2),
        ]
        for r in runtime
        if r["method"] != "keep_all"
    ]
    runtime_table = table(
        ["Method", "Median ms", "p95 ms", "Process peak MiB"], rows
    )
    manuscript = (ROOT / "manuscript.template.md").read_text()
    for token, value in {
        "DEFAULT_TABLE": default_table,
        "SENSITIVITY_TABLE": sensitivity_table,
        "CALIBRATION_TABLE": calibration_table,
        "RUNTIME_TABLE": runtime_table,
    }.items():
        manuscript = manuscript.replace("{{" + token + "}}", value)
    if "{{" in manuscript:
        raise ValueError("Unresolved manuscript placeholder")
    return manuscript


def bundle_review() -> None:
    """Archive paper sources, excluding all raw comparison data."""
    names = [
        "manuscript.tex",
        "references.bib",
        "figures/default-results.pdf",
        "figures/calibration-transfer.pdf",
    ]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in names:
            raw = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    (ROOT / "review-source.tar.gz").write_bytes(
        gzip.compress(stream.getvalue(), mtime=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    (ROOT / "manuscript.md").write_text(render(), encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            "manuscript.md",
            "--citeproc",
            "--standalone",
            "--to",
            "latex",
            "--output",
            "manuscript.tex",
        ],
        cwd=ROOT,
        check=True,
    )
    if not args.skip_pdf:
        subprocess.run(["tectonic", "manuscript.tex"], cwd=ROOT, check=True)
    bundle_review()


if __name__ == "__main__":
    main()
