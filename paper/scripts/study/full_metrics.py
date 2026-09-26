"""Document-aware summaries of the complete published-corpus comparison."""

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import ROOT, write_json
from .metrics import bootstrap_difference, wilson


def majority(flags: list) -> int:
    if not flags:
        raise ValueError("Empty document")
    return int(2 * sum(flags) >= len(flags))


def decision(row: dict, method: str) -> int:
    return (
        row["hf"][method]
        if method.startswith("hf_")
        else row["pygarble"][method]["flag"]
    )


def document_summary(doc: dict, rows: list, method: str) -> dict:
    flags = [decision(r, method) for r in rows]
    full = [decision(r, method) for r in rows if r["characters"] == 400]
    return {
        **{
            k: doc[k]
            for k in (
                "doc_id",
                "member",
                "label",
                "scope",
                "language",
                "era",
                "genre",
            )
        },
        "method": method,
        "chunks": len(flags),
        "flagged": sum(flags),
        "flagged_fraction": sum(flags) / len(flags),
        "majority_decision": majority(flags),
        "any_decision": int(any(flags)),
        "full_width_chunks": len(full),
        "tail_chunks": len(flags) - len(full),
        "full_width_fraction": sum(full) / len(full) if full else None,
        "full_width_majority": majority(full) if full else None,
    }


def summarize_documents(doc_rows: list) -> dict:
    from .full_corpus import METHODS

    lookup = {(r["doc_id"], r["method"]): r for r in doc_rows}
    results = []
    for method in METHODS:
        positives = sorted(
            [r for r in doc_rows if r["method"] == method and r["label"]],
            key=lambda r: r["doc_id"],
        )
        controls = [
            r
            for r in doc_rows
            if r["method"] == method and r["scope"] == "english"
        ]
        other = [
            r
            for r in doc_rows
            if r["method"] == method and r["scope"] == "other_language"
        ]
        tp = sum(r["majority_decision"] for r in positives)
        results.append(
            {
                "method": method,
                "positive_documents": len(positives),
                "positive_majority_detected": tp,
                "document_recall": tp / len(positives),
                "recall_wilson95_conditional": wilson(tp, len(positives)),
                "paired_document_delta_vs_hf_ci95": bootstrap_difference(
                    [r["majority_decision"] for r in positives],
                    [
                        lookup[(r["doc_id"], "hf_all")]["majority_decision"]
                        for r in positives
                    ],
                ),
                "positive_any_detected": sum(
                    r["any_decision"] for r in positives
                ),
                "positive_macro_flagged_fraction": sum(
                    r["flagged_fraction"] for r in positives
                )
                / len(positives),
                "positive_chunks": sum(r["chunks"] for r in positives),
                "positive_flagged_chunks": sum(
                    r["flagged"] for r in positives
                ),
                "english_documents": len(controls),
                "english_majority_false_flags": sum(
                    r["majority_decision"] for r in controls
                ),
                "english_any_false_flags": sum(
                    r["any_decision"] for r in controls
                ),
                "english_chunks": sum(r["chunks"] for r in controls),
                "english_false_flag_chunks": sum(
                    r["flagged"] for r in controls
                ),
                "english_macro_chunk_fpr": sum(
                    r["flagged_fraction"] for r in controls
                )
                / len(controls),
                "positive_full_width_macro_fraction": sum(
                    r["full_width_fraction"] for r in positives
                )
                / len(positives),
                "english_full_width_macro_fpr": sum(
                    r["full_width_fraction"] for r in controls
                )
                / len(controls),
                "other_language_documents": len(other),
                "other_language_majority_false_flags": sum(
                    r["majority_decision"] for r in other
                ),
                "other_language_macro_fpr": (
                    sum(r["flagged_fraction"] for r in other) / len(other)
                    if other
                    else None
                ),
            }
        )
    languages = []
    for method in METHODS:
        groups = defaultdict(list)
        for row in doc_rows:
            if row["method"] == method and not row["label"]:
                groups[row["language"]].append(row)
        for language, values in sorted(groups.items()):
            languages.append(
                {
                    "method": method,
                    "language": language,
                    "documents": len(values),
                    "chunks": sum(r["chunks"] for r in values),
                    "majority_false_flags": sum(
                        r["majority_decision"] for r in values
                    ),
                    "macro_fpr": sum(r["flagged_fraction"] for r in values)
                    / len(values),
                }
            )
    return {"primary": results, "by_language": languages}


def analyze(output: Path) -> None:
    from .full_corpus import METHODS, read_parts

    progress = json.loads((output / "progress.json").read_text())
    if not progress["complete"]:
        raise ValueError("Full corpus has not finished")
    docs = json.loads((output / "documents.json").read_text())
    rows = []
    hashes = Counter()
    tokens = Counter()
    classes = Counter()
    for doc in docs:
        predictions = read_parts(
            output, progress["documents"][doc["doc_id"]]["parts"]
        )
        if len(predictions) != doc["chunk_count"]:
            raise ValueError("Incomplete prediction coverage")
        end = 0
        for pred in predictions:
            if (
                pred["start"] != end
                or pred["stop"] - pred["start"] != pred["characters"]
            ):
                raise ValueError("Chunk gap or overlap")
            end = pred["stop"]
            hashes[pred["text_sha256"]] += 1
            tokens[pred["hf"]["tokens"]] += 1
            classes[(doc["scope"], pred["hf"]["label"])] += 1
        if end != doc["characters"]:
            raise ValueError("Unprocessed source text")
        rows.extend(
            document_summary(doc, predictions, method) for method in METHODS
        )
    (output / "document-results.json.gz").write_bytes(
        gzip.compress(json.dumps(rows, sort_keys=True).encode(), mtime=0)
    )
    summary = summarize_documents(rows)
    write_json(output / "summary.json", summary)
    preliminary = json.loads((ROOT / "results/records.json").read_text())
    primary_hashes = {
        r["text_sha256"] for r in preliminary if r["view"] == 400
    }
    write_json(
        output / "coverage.json",
        {
            "documents": len(docs),
            "meaningful_documents": sum(not d["label"] for d in docs),
            "positive_documents": sum(d["label"] for d in docs),
            "english_control_documents": sum(
                d["scope"] == "english" for d in docs
            ),
            "normalized_characters": sum(d["characters"] for d in docs),
            "chunks": sum(d["chunk_count"] for d in docs),
            "unique_chunk_hashes": len(hashes),
            "duplicate_occurrences_beyond_first": sum(
                n - 1 for n in hashes.values()
            ),
            "preliminary_primary_unique_hash_overlap": len(
                primary_hashes & hashes.keys()
            ),
            "preliminary_primary_unique_hashes": len(primary_hashes),
            "max_tokens": max(tokens),
            "truncated_inputs": 0,
            "hf_class_counts": {
                scope + ":" + label: n
                for (scope, label), n in sorted(classes.items())
            },
            "tail_chunks": sum(d["characters"] % 400 != 0 for d in docs),
            "new_human_label_audit": False,
        },
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / "document-results.csv").write_text(text.getvalue())
    lines = [
        "# Full published-corpus comparison",
        "",
        "All released labelled texts are covered; no new human label audit.",
        "English is primary. Other languages are a separate scope diagnostic.",
        "A document is flagged when at least half of its chunks are flagged.",
        "",
        "| Method | Positive documents detected | English documents falsely "
        "flagged | Mean positive chunk fraction | Mean English chunk FPR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary["primary"]:
        lines.append(
            "| {} | {}/{} | {}/{} | {:.1%} | {:.2%} |".format(
                r["method"],
                r["positive_majority_detected"],
                r["positive_documents"],
                r["english_majority_false_flags"],
                r["english_documents"],
                r["positive_macro_flagged_fraction"],
                r["english_macro_chunk_fpr"],
            )
        )
    lines += [
        "",
        "Rates average documents equally; micro counts are saved separately.",
        "Do not treat chunks as independent observations or the corpus's"
        " class",
        "balance as deployment prevalence. HF training overlap is unknown.",
        "Both HF policies use the same model: all non-clean winners"
        " (primary),",
        "or noise/word-salad winners only (secondary).",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    analyze(ROOT / "full-results")
