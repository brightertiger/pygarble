# pygarble paper and benchmark

Author: Ujjwal Singh Rao

This directory holds the evidence behind pygarble's gibberish benchmark.

- `study/`: pinned protocols, deviations, validation notes, frozen results
  (`full-results/`), saved predictions and data provenance.
- `scripts/`: study runners, analysis, study tests and the package-data and
  documentation maintenance scripts.
- `regression/`: fast engineering fixtures and golden checks for the library.

Start with the [study instructions](study/README.md), the
[full results](study/full-results/report.md) and the
[validation notes](study/validation.md). Run `make benchmark-check` for the
offline checks.

## Status

The earlier JOSS draft and preprint manuscript were removed on 2026-10-11 for
a rework. They remain in git history, in the commit that removed them. The
study protocols and frozen results here are unchanged.
