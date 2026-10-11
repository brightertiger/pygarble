# Preprint benchmark

Code for the benchmark in `paper/PREPRINT-PLAN.md`. The frozen protocol is
`paper/benchmark/protocol.md`, departures are in
`paper/benchmark/deviations.md`, and results are in
`paper/benchmark/results/`. No dataset text is committed: the manifest and
predictions hold row ids, hashes, scores and decisions only.

## Check the committed results

Needs Python 3.12 with numpy and this checkout of pygarble. It does not
need the raw data or the baselines.

```
make benchmark-verify PYTHON=/path/to/venv/bin/python
```

`verify.py` recomputes `metrics.json` and `pr_curves.json` from
`results/predictions/` and `manifest.json` and exits non-zero on any
mismatch.

## Reproduce from the raw files

1. Download the raw data into one directory, called `DATA` below:
   - `DATA/kaggle-gibberish/Gibberish.xls` from
     https://www.kaggle.com/datasets/johnwdata/gibberish-text-classification
   - `DATA/hf-sowmya15/{train,validation,test}.csv` from
     https://huggingface.co/datasets/Sowmya15/gibberish_multilingual
2. Download the baseline assets. Their sha256 hashes are checked by
   `run.py`:

   ```
   mkdir -p DATA/baselines
   curl -L -o DATA/baselines/big.txt https://raw.githubusercontent.com/domanchi/gibberish-detector/ecac96918e72fbf702cfee3274dbf280009171ad/examples/big.txt
   curl -L -o DATA/baselines/ldnoobw_en.txt https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/5faf2ba42d7b1c0977169ec3611df25a3c08eb13/en
   ```

3. Create a venv outside the repository and install the baselines and
   pygarble:

   ```
   python3.12 -m venv DATA/venv
   DATA/venv/bin/pip install -r paper/scripts/benchmark/requirements.txt
   DATA/venv/bin/pip install -e .
   ```

4. From the repository root, run each step with `DATA/venv/bin/python -m`:

   | Module | Writes |
   | --- | --- |
   | `paper.scripts.benchmark.data --data-root DATA` | `manifest.json` (checks hashes, rebuilds the split) |
   | `paper.scripts.benchmark.run --data-root DATA` | predictions, `checkpoint.json`, `metrics.json`, `pr_curves.json`, `results.md`, `environment.json` |
   | `paper.scripts.benchmark.calibrate --data-root DATA` | `calibration.json` |
   | `paper.scripts.benchmark.footprint --data-root DATA` | `footprint.json` |

`run.py` refuses to score B test a second time. To repeat the run, delete
`results/predictions/B_test/` first. The checkpoint failed in the recorded
run, so the recorded command was

```
run --override-checkpoint "<reason>"
```

See `deviations.md`. `run --metrics-only` recomputes the metrics and
report from saved predictions without scoring.

Scores are deterministic, so a rerun reproduces the predictions exactly.
Footprint timings depend on the machine; the recorded machine is in
`environment.json` and `footprint.json`.

## Files

- `data.py`: the A and B loaders, hash checks, the B split and the manifest.
- `methods.py`: the score and decision wrappers for the pygarble profiles,
  voting modes and strategies, and for the baselines.
- `rules.py`: the Gopher and C4 rules on RedPajama-V2 signal definitions.
- `metrics.py`: the cuts, metrics, bootstrap and PR curves, shared by
  `run.py` and `verify.py`.
- `report.py`: renders `results.md`.
- `test_benchmark.py`: one behaviour test per script.
