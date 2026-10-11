# pygarble paper and benchmark

Authors: Ujjwal Singh Rao and Ricky Singh

This directory holds the preprint and the evidence behind it.

- `preprint/`: the preprint, *How Much Junk Do Rules Catch?*, as LaTeX source
  (`main.tex`) and a built PDF (`main.pdf`). Its tables and number macros are
  generated from the benchmark results.
- `benchmark/`: the frozen protocol, dated deviations, the dataset manifest
  (hashes and row ids, no text) and the results, including per-row
  predictions for every method.
- `scripts/benchmark/`: the code that builds the manifest, scores the
  methods, recomputes the metrics and draws the preprint figure and tables.
- `scripts/`: the package-data and documentation maintenance scripts.
- `regression/`: fast engineering fixtures and golden checks for the library.
  They are not a research benchmark.

Start with the [preprint PDF](preprint/main.pdf), the
[protocol](benchmark/protocol.md) and the [results](benchmark/results/results.md).
From the repository root, `make benchmark-verify` recomputes every metric from
the saved predictions and `make preprint` rebuilds the PDF. Re-scoring from
the raw data is described in the [benchmark README](scripts/benchmark/README.md).
The raw data is never committed.

## Status

The 2026-09 study of the Gaskell–Bowern corpus with a DistilBERT comparison
was retired on 2026-10-11 and deleted from the tree. It remains in git
history; the last commit that contains it is f545b0d.
