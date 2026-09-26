# Engineering regression checks

The canonical research benchmark is [the paper's published-corpus study](../paper/study/README.md).
This directory keeps fast development fixtures and behavior checks; its authored
examples do not measure general-world accuracy.

- `english_challenge.json`, checksum, `benchmark_data.json` and
  `label_overrides.json`: development inputs and reviewed legacy labels.
  `evaluate.py` reports their errors without changing the historical labels.
- `golden.py`, `golden.jsonl` and checksum: expected gibberish API outputs.
- `golden_scan.py`, `golden_scan.jsonl` and checksum: expected scanner outputs.
- `scan_vectors.json` and `clean_corpus/`: screening rules and meaningful text
  that should not trigger high-confidence findings.
- `throughput.py`: synthetic scanner throughput, separate from the paper.

```bash
python regression/evaluate.py --split development
python regression/golden.py --check
python regression/golden_scan.py --check
python regression/throughput.py --size-mb 2
```

Regenerate golden outputs only for intended behavior changes and review the
diff. Store optional local reports outside tracked files. Superseded aggregate
benchmark reports, the old benchmark runner and exploratory trigram/runtime
scripts remain recoverable from Git history.
