# Engineering regression checks

The research benchmark is [the preprint benchmark](../benchmark/protocol.md).
This directory keeps fast behavior checks for the library; they catch
regressions and keep ports comparable, and do not measure accuracy.

- `golden_inputs.json`: 600 rows sampled with seed 20261011 from the English
  rows of the preprint benchmark's open dataset, Hugging Face
  Sowmya15/gibberish_multilingual (Apache-2.0): 300 random strings (label 1)
  and 300 tweets (label 0), each with the benchmark's row id. Its
  `provenance` block names the source files, their sha256 and the sampling
  rule. `sample_golden_inputs.py` drew it and needs the raw dataset; nothing
  else does. No text from the Kaggle dataset is committed.
- `golden.py`, `golden.jsonl` and checksum: expected gibberish API outputs
  (garbled, score, status and spans) under every profile, for the golden
  inputs followed by the hand-written edge inputs in `golden.py`.
- `golden_scan.py`, `golden_scan.jsonl` and checksum: expected scanner outputs
  over the scan vectors, the clean corpus and the same gibberish inputs.
- `scan_vectors.json` and `clean_corpus/`: screening rules and meaningful text
  that should not trigger high-confidence findings.
- `throughput.py`: synthetic scanner throughput, separate from the paper.

```bash
python paper/regression/golden.py --check
python paper/regression/golden_scan.py --check
python paper/regression/throughput.py --size-mb 2
```

Regenerate golden outputs only for intended behavior changes and review the
diff. Store optional local reports outside tracked files.

On 2026-10-11 the golden inputs changed. They were the 132 hand-written cases
of the old authored challenge set (`english_challenge.json`, with
`benchmark_data.json`, `label_overrides.json` and `evaluate.py`). That set was
replaced by the preprint benchmark, so the golden corpus now samples the
benchmark's open dataset instead; the edge inputs and the row format are
unchanged, and the old corpus of 1,071 rows became 4,375 rows (625 texts under
7 profiles). The old challenge set, superseded benchmark reports and runners
remain recoverable from Git history.
