"""Optional, pinned local transformer benchmark; no hosted inference."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import List

from .data import ROOT

LABELS = ("clean", "mild gibberish", "noise", "word salad")


def verify_model(directory: Path, allow_download: bool = False) -> None:
    manifest = json.loads((ROOT / "hf_model.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        path = directory / entry["name"]
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(
                    "Run download-model first: " + str(path)
                )
            url = "https://huggingface.co/{}/resolve/{}/{}".format(
                manifest["model_id"], manifest["revision"], entry["name"]
            )
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=90) as response:
                with temporary.open("wb") as handle:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            temporary.replace(path)
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != entry["sha256"]:
            raise ValueError("Model checksum mismatch: " + entry["name"])


def policies(probabilities: List[float]) -> dict:
    if len(probabilities) != 4 or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Expected four class probabilities")
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Class probabilities must sum to one")
    winner = max(range(4), key=probabilities.__getitem__)
    return {
        "label": LABELS[winner],
        "hf_all": int(winner != 0),
        "hf_strict": int(winner in (2, 3)),
        "score": 1 - probabilities[0],
    }


class HFModel:
    def __init__(self, directory: Path, threads: int = 4) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260927)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                attn_implementation="sdpa",
            )
            .to("cpu")
            .eval()
        )
        if tuple(self.model.config.id2label[i] for i in range(4)) != LABELS:
            raise ValueError("Unexpected model label ordering")
        if sum(p.numel() for p in self.model.parameters()) != 66956548:
            raise ValueError("Unexpected model parameter count")

    def predict_batch(self, texts: List[str]) -> list:
        if not texts:
            return []
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        )
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError(
                "Token limit exceeded; refusing silent truncation"
            )
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1).tolist()
        counts = inputs["attention_mask"].sum(dim=1).tolist()
        return [
            {"probabilities": p, "tokens": n, **policies(p)}
            for p, n in zip(probabilities, counts)
        ]

    def score(self, text: str) -> tuple:
        row = self.predict_batch([text])[0]
        return row["score"], True, row["label"]
