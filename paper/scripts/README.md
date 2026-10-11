# Repository and research tools

The Python files directly in this directory maintain the package and
documentation. Run them from the repository root:

```bash
python paper/scripts/update_strategy_docs.py --check
python paper/scripts/generate_data.py --check
python paper/scripts/check_discovery.py docs/_build/html
```

`data_curation.json` pins the word-frequency source, its checksum and curated
changes. `generate_data.py --source PATH --check` supports an offline source.
The generated package tables retain their historical header paths so their
bytes and recorded source hashes remain unchanged. The current
script and provenance locations are here in `paper/scripts/`.

`benchmark/` contains the preprint benchmark: data loaders, scoring, metrics,
verification and the preprint figure and tables. From the repository root,
`make benchmark-verify` recomputes the metrics from the saved predictions and
`make preprint` rebuilds the preprint. See the
[benchmark README](benchmark/README.md) for re-scoring from the raw data.
