"""Manuscript tables from the complete corpus's saved measurements."""

import gzip
import json

from .data import ROOT

NAMES = {
    "english": "English",
    "english_extended": "Extended",
    "legacy": "Legacy",
    "word_lookup": "Word lookup",
    "entropy_based": "Entropy",
    "hf_all": "HF non-clean",
    "hf_strict": "HF noise/salad",
}
CONTROL_NAMES = {
    "texts/Historical - English - Literary - NT (KJV).txt": "KJV",
    "texts/Modern - English - Literary - NT.txt": "NET",
    "texts/Modern - English - Technical - Voynich Wiki.txt": "Wiki",
    "texts/Historical - English - Technical - Secreta Alberti.txt": "Secreta",
}


def table(header: list, rows: list) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render() -> str:
    output = ROOT / "full-results"
    summary = json.loads((output / "summary.json").read_text())
    runtime = json.loads((output / "runtime.json").read_text())
    coverage = json.loads((output / "coverage.json").read_text())
    doc_rows = json.loads(
        gzip.decompress((output / "document-results.json.gz").read_bytes())
    )
    primary = {r["method"]: r for r in summary["primary"]}
    # Guard the fixed numerical prose as well as generating measured tables.
    expected = {
        "english": (13, 41),
        "english_extended": (20, 58),
        "legacy": (13, 6),
        "word_lookup": (38, 0),
        "entropy_based": (0, 0),
        "hf_all": (38, 1494),
        "hf_strict": (38, 6),
    }
    for method, (detected, false_flags) in expected.items():
        row = primary[method]
        if (
            row["positive_majority_detected"] != detected
            or row["english_false_flag_chunks"] != false_flags
            or row["english_chunks"] != 5200
        ):
            raise ValueError("Numerical manuscript prose needs review")
    for method, count in (
        ("word_lookup", 49),
        ("hf_all", 62),
        ("hf_strict", 52),
    ):
        if primary[method]["other_language_majority_false_flags"] != count:
            raise ValueError("Language diagnostic prose needs review")
    applicability = json.loads((output / "applicability.json").read_text())
    if any(r["insufficient_evidence"] for r in applicability):
        raise ValueError("Applicability prose needs review")
    if coverage["chunks"] != 79969 or coverage["documents"] != 109:
        raise ValueError("Manuscript corpus counts need review")
    rows = []
    for method, name in NAMES.items():
        row = primary[method]
        rows.append(
            [
                name,
                "{}/38".format(row["positive_majority_detected"]),
                "{:.1f}--{:.1f}".format(
                    *[100 * v for v in row["recall_wilson95_conditional"]]
                ),
                "{:.1%}".format(row["positive_macro_flagged_fraction"]),
                "{}/4".format(row["english_majority_false_flags"]),
                "{:.2%}".format(row["english_macro_chunk_fpr"]),
            ]
        )
    values = {
        "PRIMARY_TABLE": table(
            [
                "Method",
                "TP docs",
                "95% CI (%)",
                "Pos. mean",
                "FP docs",
                "Neg. mean",
            ],
            rows,
        )
    }
    controls = [
        next(
            r
            for r in doc_rows
            if r["method"] == "english" and r["member"] == member
        )
        for member in CONTROL_NAMES
    ]
    rows = []
    for method, name in NAMES.items():
        by_id = {r["doc_id"]: r for r in doc_rows if r["method"] == method}
        rows.append(
            [name]
            + [
                "{}/{}".format(by_id[d["doc_id"]]["flagged"], d["chunks"])
                for d in controls
            ]
        )
    values["CONTROL_TABLE"] = table(
        ["Method"] + list(CONTROL_NAMES.values()), rows
    )
    rows = []
    for method, name in NAMES.items():
        row = primary[method]
        rows.append(
            [
                name,
                "{}/38".format(row["positive_any_detected"]),
                "{}/4".format(row["english_any_false_flags"]),
                "{}/38".format(
                    sum(
                        r["full_width_majority"]
                        for r in doc_rows
                        if r["method"] == method and r["label"]
                    )
                ),
                "{:.1%}".format(row["positive_full_width_macro_fraction"]),
                "{:.2%}".format(row["english_full_width_macro_fpr"]),
            ]
        )
    values["TAIL_TABLE"] = table(
        [
            "Method",
            "Any TP",
            "Any FP",
            "Full TP",
            "Full pos. mean",
            "Full neg. mean",
        ],
        rows,
    )
    rows = []
    for method, name in NAMES.items():
        row = primary[method]
        rows.append(
            [
                name,
                "{}/67".format(row["other_language_majority_false_flags"]),
                "{:.1%}".format(row["other_language_macro_fpr"]),
            ]
        )
    values["LANGUAGE_TABLE"] = table(
        ["Method", "Other-language FP documents", "Mean chunk FPR"], rows
    )
    values["RUNTIME_TABLE"] = table(
        ["Method", "Median ms", "p95 ms", "Process peak MiB"],
        [
            [
                NAMES[r["method"]],
                "{:.3f}".format(r["warm_median_ms"]),
                "{:.3f}".format(r["warm_p95_ms"]),
                "{:.1f}".format(r["peak_process_rss_bytes"] / 1024**2),
            ]
            for r in runtime
        ],
    )
    times = {r["method"]: r for r in runtime}
    values["WORD_HF_RATIO"] = "{:.0f}".format(
        times["hf_all"]["warm_median_ms"]
        / times["word_lookup"]["warm_median_ms"]
    )
    values["ENGLISH_HF_RATIO"] = "{:.1f}".format(
        times["hf_all"]["warm_median_ms"] / times["english"]["warm_median_ms"]
    )
    values["DUPLICATES"] = str(coverage["duplicate_occurrences_beyond_first"])
    values["OVERLAP"] = str(
        coverage["preliminary_primary_unique_hash_overlap"]
    )
    values["MAX_TOKENS"] = str(coverage["max_tokens"])
    values["TAILS"] = str(coverage["tail_chunks"])
    values["HF_MACRO_FPR"] = "{:.2%}".format(
        primary["hf_all"]["english_macro_chunk_fpr"]
    )
    values["HF_STRICT_FPR"] = "{:.2%}".format(
        primary["hf_strict"]["english_macro_chunk_fpr"]
    )
    values["RECALL_DIFFERENCE_TABLE"] = table(
        [
            "Method minus HF non-clean",
            "Difference (pp)",
            "Paired 95% interval (pp)",
        ],
        [
            [
                NAMES[m],
                "{:.1f}".format(
                    100
                    * (
                        r["document_recall"]
                        - primary["hf_all"]["document_recall"]
                    )
                ),
                "{:.1f} to {:.1f}".format(
                    *[100 * v for v in r["paired_document_delta_vs_hf_ci95"]]
                ),
            ]
            for m, r in primary.items()
            if m != "hf_all"
        ],
    )
    manuscript = (ROOT / "manuscript.template.md").read_text()
    for token, value in values.items():
        manuscript = manuscript.replace("{{" + token + "}}", value)
    if "{{" in manuscript:
        raise ValueError("Unresolved manuscript placeholder")
    return manuscript


if __name__ == "__main__":
    (ROOT / "manuscript.md").write_text(render())
