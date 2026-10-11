"""Write golden_inputs.json: a seeded sample of the preprint's open dataset.

Draws 300 random strings (label 1) and 300 tweets (label 0) from the English
rows of Hugging Face Sowmya15/gibberish_multilingual (Apache-2.0), loaded and
row-identified exactly as the preprint benchmark loads dataset B. Only this
dataset's text is written; the Kaggle dataset is never read. Needs the raw
files under --data-root and is run once; golden.py only reads the output.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.scripts.benchmark.data import (  # noqa: E402
    DEFAULT_ROOT,
    SOURCES,
    _order,
    check_hashes,
    load_b,
)

OUTPUT = Path(__file__).resolve().parent / "golden_inputs.json"
SEED = 20261011
PER_LABEL = 300


def build(root: Path) -> dict:
    check_hashes(root)
    first = {}
    for row in sorted(load_b(root), key=lambda r: _order(r.id)):
        first.setdefault(row.text, row)
    rng = random.Random(SEED)
    picked = []
    for label in (1, 0):
        pool = [row for row in first.values() if row.label == label]
        picked.extend(rng.sample(pool, PER_LABEL))
    picked.sort(key=lambda r: _order(r.id))
    return {
        "provenance": {
            "dataset": SOURCES["B"]["name"],
            "url": SOURCES["B"]["url"],
            "licence": SOURCES["B"]["licence"],
            "files": SOURCES["B"]["files"],
            "seed": SEED,
            "sampled": (
                "English rows of train, validation and test loaded as "
                "paper/scripts/benchmark/data.py load_b does (row id "
                "'<split>:<0-based data-row position>'); duplicate texts "
                "dropped keeping the first row id; random.Random(seed).sample "
                f"of {PER_LABEL} label-1 rows (random strings) then "
                f"{PER_LABEL} label-0 rows (tweets); output sorted by row id"
            ),
            "script": "paper/regression/sample_golden_inputs.py",
        },
        "rows": [
            {"id": row.id, "text": row.text, "label": row.label}
            for row in picked
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    data = build(args.data_root)
    OUTPUT.write_text(
        json.dumps(data, ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(data['rows'])} rows to {OUTPUT.name}")


if __name__ == "__main__":
    main()
