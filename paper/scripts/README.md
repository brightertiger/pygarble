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
bytes and the paper's recorded source hashes remain unchanged. The current
script and provenance locations are here in `paper/scripts/`.

`study/` contains the research runners, analysis, study tests and paper builder.
Use `make benchmark-check`, `make benchmark-prepare`, `make benchmark` and
`make paper` from the repository root. The manuscript, pinned assets, results
and cache remain in `paper/study/`; see the
[study instructions](../study/README.md) for individual module commands and
historical reproduction.
