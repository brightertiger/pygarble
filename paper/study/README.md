# Complete published-corpus comparison

Start with [the revised software-and-evaluation PDF](manuscript.pdf),
[full results](full-results/report.md) and [publication status](../README.md#publication-status).
The main experiment evaluates every released labelled document from the
published GitHub collection: **109 documents, 31,964,664 normalized characters,
79,969 chunks**, with identical inputs for pygarble and a pinned local
Hugging Face DistilBERT classifier. All source labels are inherited; no new
human annotation or audit is performed. The authorized arXiv submission is blocked before upload by cs.CL endorsement.

The primary comparison uses all 38 gibberish documents and all text in four
English controls (5,200 negative chunks). The other 67 meaningful documents
are a separate language diagnostic. This avoids turning meaningful foreign
languages into invented positive labels or treating 80,000 correlated chunks
as independent research samples. See the [fixed extension protocol](full_corpus_protocol.md).

## Quick checks

From the repository root, `make benchmark-check` runs the 25 standard-library
study tests, including saved confusion-count verification. CI runs this without
model dependencies. `make benchmark-prepare` downloads pinned assets, and
`make benchmark` runs a new comparison to ignored `reproduction-full/` after
the optional environment is installed. These commands do not replace the
paper's frozen `full-results/`.

## Reproduce the complete comparison

Run from the repository root. The optional HF environment was measured with
Python 3.12.2; it is separate from pygarble's dependency-free base installation.
Use a dedicated environment if desired:

```bash
python3.12 -m venv paper/study/.cache/hf-venv
source paper/study/.cache/hf-venv/bin/activate
python -m pip install -r paper/study/requirements-hf.txt
python -m unittest paper.study.test_study paper.study.test_full_corpus paper.study.test_chunk_metrics -v
python -m paper.study.full_corpus --download-model
python -m paper.study.full_corpus --output paper/study/reproduction-full
```

Preparation retrieves about 14 MB of corpus archives and 268 MB of model
weights, plus tokenizer/config files. Every asset is pinned and checksum
verified. There is no hosted inference endpoint, API key or model training.
Inference runs on CPU, uses batches of eight and four PyTorch threads, and
can take substantial time on a laptop. It resumes verified completed documents
if interrupted; incomplete documents are rerun. A code-manifest mismatch stops
resumption instead of mixing implementations.

`--download-model` downloads/verifies assets and exits. The following command
runs the evaluation. Subsequent scoring uses local model files and no network
inference. The source cache is also verified on reuse. An input exceeding the
model's token limit would stop the run instead of truncating silently.

To validate the checked-in results after preparing source/model assets:

```bash
python -m paper.study.full_verify --replay
python -m paper.study.full_diagnostics
```

`full_verify` checks all saved input identities, probabilities, decisions,
coverage, code hashes and summaries. `--replay` additionally scores first,
middle and last chunks from every document in a new offline model process.
This is a fixed subset replay, not a second complete neural run. Replay allows
probability differences up to 1e-5 from batch padding but requires identical
decisions and exact pygarble scores/public-API decisions.

For a new output directory, pass `--output` to `full_verify`. Recompute saved
summaries without model inference using `full_corpus --analyze-only --output ...`.
Do not run CPU benchmarks concurrently with the exhaustive model evaluation:

```bash
python -m paper.study.full_runtime
```

This re-measures five pygarble configurations and the shared HF model in
isolated timing and memory workers, using one thread, batch size one and the
same fixed 76 inputs. It writes `full-results/runtime.json`. Local costs include
CPU time, memory and disk; no external inference/compute fee is incurred.

## Paper and figures

```bash
python -m pip install -r paper/study/requirements-figures.txt
python -m paper.study.chunk_metrics
python -m paper.study.full_figures
# Requires Pandoc and Tectonic on PATH:
python -m paper.study.build_paper
```

Edit `manuscript.template.md`; `full_paper.py` inserts measured tables and
numerical placeholders. `build_paper.py` generates Markdown, TeX and the review
source archive under ignored `.cache/publication/`, and updates the tracked
`manuscript.pdf` for review. Only the two referenced figure PDFs are retained
in Git; optional figure previews can be regenerated. See [validation](validation.md) for executed checks.
`article.tex` supplies the two-column article layout; `tables.lua` renders
numbered table floats. The title and body foreground pygarble's architecture
and detection strategies before presenting the comparison. Wide appendix
material uses the full page. The byline contains only Ujjwal Singh Rao.

Chunk accuracy, precision, recall, F1 and confusion matrices are additional
post hoc summaries of saved predictions. The original source-level endpoints
remain in Appendix A; no scoring settings changed. `chunk_metrics.py` writes
`full-results/chunk-metrics.json`, and `test_chunk_metrics.py` independently
recounts all target-comparison predictions.

The separate JOSS-format software draft is `../paper.md`. Funding, conflicts,
Claude's assistance and the arXiv license choice have been confirmed. The
arXiv workflow awaits category endorsement; see the publication status.

## Artifact map

- `full_corpus_protocol.md`: fixed extension methods and limitations.
- `sources.json`, `hf_model.json`, `LICENSES.md`: source/model provenance,
  immutable revisions, checksums and separate rights.
- `full_corpus.py`, `hf_backend.py`: exhaustive chunking, package adapters,
  offline model inference and checkpointed per-document predictions.
- `full_metrics.py`: document aggregation, source-aware rates and intervals.
- `full-results/`: all measured full-corpus artifacts. Predictions are small
  compressed JSONL shards with input hashes and offsets, not source prose.
- `full-results/document-results.csv` and `.json.gz`: every method/document
  decision, chunk rate, any-chunk result and tail-excluded rate.
- `full-results/summary.json`: primary and per-language summaries.
- `full-results/applicability.json`, `disagreements.json`: supplementary
  abstention coverage and descriptive paired chunk counts.
- `full_verify.py`, `test_full_corpus.py`: artifact/replay verification and tests.
- `full_runtime.py`: isolated CPU/RSS measurement.
- `full_figures.py`, `full_paper.py`, `build_paper.py`: publication artifacts.

Raw comparison text and model weights remain in ignored `.cache/`; their
separate licenses still apply. No package runtime code or dependency changes
are required by this study.

## Retained preliminary experiment

`results/`, `protocol.md`, `data.py`, `detectors.py`, `metrics.py`, `run.py`,
`runtime.py` and `verify_results.py` retain the earlier capped experiment and
its frozen code. That experiment used 38 positive prefixes and 400 English
control blocks, additional length views and source-held-out calibration of
character models and package thresholds. It is preliminary and overlaps the
full evaluation. Its character-model calibration is not represented as a
new full-corpus run.

```bash
python -m paper.study.run --output paper/study/reproduction --skip-timing
python -m paper.study.verify_results
```

This preliminary runner uses only the standard library and local pygarble.
A fresh dependency-free virtual environment reproduced its deterministic
artifacts byte for byte. That evidence is distinct from the full transformer
run's exhaustive artifact checks and fixed subset replay. See
[the original results](results/report.md) and [protocol deviations](deviations.md).
