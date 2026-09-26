"""Supplementary coverage and pairwise disagreements, without label tuning."""

import json
from collections import defaultdict

from .data import ROOT, write_json
from .full_corpus import PACKAGE_METHODS, read_parts


def main() -> None:
    output = ROOT / "full-results"
    progress = json.loads((output / "progress.json").read_text())
    if not progress["complete"]:
        raise ValueError("Run complete evaluation first")
    docs = json.loads((output / "documents.json").read_text())
    coverage = []
    hash_labels = defaultdict(set)
    hash_sources = defaultdict(set)
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for doc in docs:
        rows = read_parts(
            output, progress["documents"][doc["doc_id"]]["parts"]
        )
        for row in rows:
            hash_labels[row["text_sha256"]].add(doc["label"])
            hash_sources[row["text_sha256"]].add(doc["doc_id"])
        for method in PACKAGE_METHODS:
            coverage.append(
                {
                    "doc_id": doc["doc_id"],
                    "scope": doc["scope"],
                    "language": doc["language"],
                    "method": method,
                    "chunks": len(rows),
                    "insufficient_evidence": sum(
                        not row["pygarble"][method]["applicable"]
                        for row in rows
                    ),
                }
            )
            for policy in ("hf_all", "hf_strict"):
                cells = groups[(doc["scope"], method, policy)]
                for row in rows:
                    a = row["pygarble"][method]["flag"]
                    b = row["hf"][policy]
                    cells[a * 2 + b] += 1
    write_json(
        output / "duplicate-diagnostics.json",
        {
            "chunk_hashes_with_conflicting_inherited_labels": sorted(
                h for h, labels in hash_labels.items() if len(labels) > 1
            ),
            "chunk_hashes_present_in_multiple_documents": {
                h: sorted(sources)
                for h, sources in sorted(hash_sources.items())
                if len(sources) > 1
            },
            "note": "Exact normalized chunk matches, not semantic overlap",
        },
    )
    write_json(output / "applicability.json", coverage)
    write_json(
        output / "disagreements.json",
        [
            {
                "scope": scope,
                "method": method,
                "reference": policy,
                "neither_flags": counts[0],
                "hf_only_flags": counts[1],
                "pygarble_only_flags": counts[2],
                "both_flag": counts[3],
                "unit": "descriptive dependent chunk counts",
            }
            for (scope, method, policy), counts in sorted(groups.items())
        ],
    )


if __name__ == "__main__":
    main()
