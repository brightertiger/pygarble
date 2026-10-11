"""Load benchmark datasets A and B and write the manifest.

Only row ids, hashes and counts are written to the repository; the texts
stay in the raw files under --data-root.
"""

import argparse
import csv
import hashlib
import io
import json
import statistics
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

DEFAULT_ROOT = Path("/Users/ujjwal/Downloads/solo/pygarble-data")
REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT = REPOSITORY / "paper" / "benchmark"
MANIFEST = OUTPUT / "manifest.json"
SEED = 20261011
A_FILE = "kaggle-gibberish/Gibberish.xls"
B_SPLITS = ("train", "validation", "test")
MOJIBAKE = ("ð\x9f", "â\x80", "Ã")
SOURCES = {
    "A": {
        "name": "Kaggle johnwdata/gibberish-text-classification",
        "url": "https://www.kaggle.com/datasets/johnwdata/"
        "gibberish-text-classification",
        "licence": "Unknown (as listed on Kaggle)",
        "files": {
            A_FILE: "091d5d3895f035d5cadaf505ac1ecb43"
            "beb675125fd2b19f0de9a5e6c85882a4"
        },
    },
    "B": {
        "name": "Hugging Face Sowmya15/gibberish_multilingual",
        "url": "https://huggingface.co/datasets/Sowmya15/"
        "gibberish_multilingual",
        "licence": "Apache-2.0",
        "files": {
            "hf-sowmya15/train.csv": "33c1329308672f9d3dc93cbf942be4e2"
            "553875e67be1175937653e06846bd12e",
            "hf-sowmya15/validation.csv": "3f1b1e860b935e2056b242abceffd425"
            "b2d3bdf7ab2e2bc6b648d5e2be5f797d",
            "hf-sowmya15/test.csv": "22b27b9c0f43d7beb3ef24c048f936b8"
            "253e17a49cbcaf75aa6105b3cf23a4b6",
        },
    },
}


class Row(NamedTuple):
    id: str
    text: str
    label: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(root: Path) -> None:
    for source in SOURCES.values():
        for name, expected in source["files"].items():  # type: ignore
            actual = sha256(root / name)
            if actual != expected:
                raise ValueError(f"{name}: sha256 {actual} != {expected}")


def load_a(root: Path = DEFAULT_ROOT) -> List[Row]:
    """Dataset A: every row is junk. Strip, drop empty, keep first copy.

    The file is a Windows-1252 CSV despite its .xls extension. Row ids are
    0-based data-row positions in the file.
    """
    raw = (root / A_FILE).read_bytes().decode("cp1252")
    reader = csv.reader(io.StringIO(raw, newline=""))
    if next(reader) != ["Response", "Label"]:
        raise ValueError("unexpected header in dataset A")
    seen = set()
    rows = []
    for position, (response, label) in enumerate(reader):
        if label != "1":
            raise ValueError(f"A row {position}: label {label!r}")
        text = response.strip()
        if text and text not in seen:
            seen.add(text)
            rows.append(Row(f"a:{position}", text, 1))
    return rows


def load_b(root: Path = DEFAULT_ROOT) -> List[Row]:
    """Dataset B: English rows of all three splits, labels as given.

    Row ids are "<split>:<0-based data-row position>". Texts are kept
    exactly as given; rows whose text is empty after strip are dropped.
    """
    rows = []
    for split in B_SPLITS:
        path = root / "hf-sowmya15" / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for position, record in enumerate(csv.DictReader(handle)):
                if record["language"].strip().lower() != "english":
                    continue
                if not record["text"].strip():
                    continue
                label = int(record["label"])
                if label not in (0, 1):
                    raise ValueError(f"B {split}:{position}: label {label}")
                rows.append(Row(f"{split}:{position}", record["text"], label))
    return rows


def _order(row_id: str) -> Tuple[int, int]:
    split, position = row_id.split(":")
    return B_SPLITS.index(split), int(position)


def build_manifest(root: Path = DEFAULT_ROOT) -> Dict:
    check_hashes(root)
    a_rows = load_a(root)
    b_rows = load_b(root)
    labels = {row.id: row.label for row in b_rows}
    texts = {row.id: row.text for row in b_rows}
    tweets = [row.text for row in b_rows if row.label == 0]
    languages = set()
    for split in B_SPLITS:
        path = root / "hf-sowmya15" / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            languages.update(
                record["language"].strip().lower()
                for record in csv.DictReader(handle)
            )
    sources = json.loads(json.dumps(SOURCES))
    sources["A"]["rows_in_file"] = 3767
    sources["A"]["kept"] = len(a_rows)
    sources["B"]["kept"] = len(b_rows)
    return {
        "frozen": "2026-10-11",
        "seed": SEED,
        "sources": sources,
        "counts": {
            "A": len(a_rows),
            "B": len(b_rows),
            "B_positive": sum(labels.values()),
            "B_negative": len(b_rows) - sum(labels.values()),
        },
        "notes": {
            "B_tweets_with_mojibake_share": round(
                sum(any(m in t for m in MOJIBAKE) for t in tweets)
                / len(tweets),
                4,
            ),
            "B_mojibake_signatures": ["U+00F0 0x9F", "U+00E2 0x80", "U+00C3"],
            "B_duplicate_text_rows": len(b_rows) - len(set(texts.values())),
            "A_median_chars": statistics.median(len(r.text) for r in a_rows),
            "A_single_token_fraction": round(
                sum(len(r.text.split()) == 1 for r in a_rows) / len(a_rows), 4
            ),
            "B_languages": len(languages),
        },
        "A_ids": [row.id for row in a_rows],
        "B_ids": sorted(labels, key=_order),
        "B_labels": {i: labels[i] for i in sorted(labels, key=_order)},
    }


def load(root: Path = DEFAULT_ROOT) -> Dict[str, List[Row]]:
    """Rows for A and B in manifest order, checked against the manifest."""
    manifest = json.loads(MANIFEST.read_text())
    check_hashes(root)
    a_rows = load_a(root)
    if [row.id for row in a_rows] != manifest["A_ids"]:
        raise ValueError("dataset A does not match the manifest")
    b_rows = load_b(root)
    if [row.id for row in b_rows] != manifest["B_ids"]:
        raise ValueError("dataset B does not match the manifest")
    if any(manifest["B_labels"][row.id] != row.label for row in b_rows):
        raise ValueError("dataset B labels do not match the manifest")
    return {"A": a_rows, "B": b_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["counts"]), json.dumps(manifest["notes"]))


if __name__ == "__main__":
    main()
