"""Retrieve pinned published data; retain labels without relabeling."""

import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .paths import ROOT

FAMILIES = ("bible", "secreta", "wiki")
LENGTHS = (100, 400, 800)
ENGLISH = {
    "texts/Historical - English - Literary - NT (KJV).txt": (
        "kjv",
        "bible",
        "historical",
    ),
    "texts/Historical - English - Technical - Secreta Alberti.txt": (
        "secreta",
        "secreta",
        "historical",
    ),
    "texts/Modern - English - Literary - NT.txt": ("net", "bible", "modern"),
    "texts/Modern - English - Technical - Voynich Wiki.txt": (
        "wiki",
        "wiki",
        "modern",
    ),
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def download(cache: Path) -> None:
    """Verify cached files too; never silently accept a changed archive."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "sources.json").read_text())
    for item in manifest["files"]:
        path = cache / item["filename"]
        if not path.exists():
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                raw = response.read()
            if digest(raw) != item["sha256"]:
                raise ValueError("Download checksum mismatch: " + path.name)
            path.write_bytes(raw)
        if digest(path.read_bytes()) != item["sha256"]:
            raise ValueError("Cached checksum mismatch: " + path.name)


def documents(cache: Path) -> List[Dict[str, Any]]:
    result = []
    for archive, label in [
        ("gibberish_transcriptions.zip", 1),
        ("meaningful.zip", 0),
    ]:
        with zipfile.ZipFile(cache / archive) as handle:
            names = sorted(handle.namelist())
            for name in names:
                if label:
                    if not re.fullmatch(r"Gibberish - D[AC]_\d+\.txt", name):
                        continue
                    doc_id = Path(name).stem.replace("Gibberish - ", "")
                    family, domain = "gibberish", "human_produced"
                elif name in ENGLISH:
                    doc_id, family, domain = ENGLISH[name]
                else:
                    continue
                raw = handle.read(name)
                text = " ".join(raw.decode("utf-8-sig").split())
                result.append(
                    {
                        "doc_id": doc_id,
                        "family": family,
                        "domain": domain,
                        "label": label,
                        "archive": archive,
                        "member": name,
                        "raw_sha256": digest(raw),
                        "text": text,
                        "normalized_sha256": digest(text.encode("utf-8")),
                        "characters": len(text),
                    }
                )
    counts = Counter(doc["label"] for doc in result)
    if counts != {1: 38, 0: 4}:
        raise ValueError("Unexpected source membership: " + str(counts))
    return result


def block_starts(length: int, width: int, limit: int = 100) -> List[int]:
    count = length // width
    if count <= limit:
        return [i * width for i in range(count)]
    return [int(i * (count - 1) / (limit - 1)) * width for i in range(limit)]


def samples(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for doc in docs:
        text = doc["text"]
        widths = list(LENGTHS) + ([0] if doc["label"] else [])
        for width in widths:
            size = width or len(text)
            if len(text) < size:
                continue
            starts = [0] if doc["label"] else block_starts(len(text), size)
            for start in starts:
                part = text[start : start + size]
                result.append(
                    {
                        "id": "{}:{}:{}".format(doc["doc_id"], width, start),
                        "doc_id": doc["doc_id"],
                        "family": doc["family"],
                        "domain": doc["domain"],
                        "label": doc["label"],
                        "view": width,
                        "start": start,
                        "stop": start + size,
                        "characters": len(part),
                        "utf8_bytes": len(part.encode()),
                        "text_sha256": digest(part.encode()),
                        "text": part,
                    }
                )
    return result


def shingles(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def overlap_report(
    docs: List[Dict[str, Any]], rows: List[Dict[str, Any]]
) -> dict:
    controls = [doc for doc in docs if not doc["label"]]
    pairwise = []
    sets = {doc["doc_id"]: shingles(doc["text"]) for doc in controls}
    for index, a in enumerate(controls):
        for b in controls[index + 1 :]:
            sa, sb = sets[a["doc_id"]], sets[b["doc_id"]]
            pairwise.append(
                {
                    "a": a["doc_id"],
                    "b": b["doc_id"],
                    "same_family": a["family"] == b["family"],
                    "jaccard": len(sa & sb) / len(sa | sb),
                    "smaller_containment": len(sa & sb)
                    / min(len(sa), len(sb)),
                }
            )
    primary = [row for row in rows if row["view"] == 400]
    hashes = Counter(row["text_sha256"] for row in primary)

    # Recursive string extraction covers existing fixture schemas without
    # assigning their labels to this new dataset.
    def strings(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        return []

    old_strings = []
    old_files = []
    for path in sorted((ROOT.parent / "regression").glob("*.json")):
        old_files.append(
            {"file": path.name, "sha256": digest(path.read_bytes())}
        )
        old_strings.extend(strings(json.loads(path.read_text())))
    old_texts = {" ".join(text.split()) for text in old_strings}
    old_shingles = set().union(*(shingles(t) for t in old_texts))
    matches = []
    for row in primary:
        ss = shingles(row["text"])
        containment = len(ss & old_shingles) / len(ss) if ss else 0.0
        if row["text"] in old_texts or containment >= 0.5:
            matches.append({"id": row["id"], "containment": containment})
    return {
        "control_pairs": pairwise,
        "duplicate_primary_hashes": {k: n for k, n in hashes.items() if n > 1},
        "development_files": old_files,
        "development_overlap_at_least_half_shingles": matches,
        "warning": "Automated overlap checks do not validate source labels.",
    }


def prepare(cache: Path, output: Path) -> Tuple[list, list]:
    download(cache)
    docs = documents(cache)
    rows = samples(docs)
    write_json(
        output / "documents.json",
        [{k: v for k, v in doc.items() if k != "text"} for doc in docs],
    )
    write_json(
        output / "records.json",
        [{k: v for k, v in row.items() if k != "text"} for row in rows],
    )
    overlap = overlap_report(docs, rows)
    write_json(output / "overlap.json", overlap)
    for pair in overlap["control_pairs"]:
        if not pair["same_family"] and pair["smaller_containment"] > 0.5:
            raise ValueError("Severe cross-family overlap; review protocol")
    return docs, rows
