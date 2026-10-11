# Preprint benchmark

Code for the preprint benchmark. The frozen protocol is
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

`verify.py` recomputes `metrics.json` from `results/predictions/` and
`manifest.json`, and `tuning/tuning.json` from the cached strategy scores,
the recorded tuning row ids and the learned weights (Optuna is not re-run).
It exits non-zero on any mismatch.

## Reproduce from the raw files

1. Download the raw data into one directory, called `DATA` below:
   - `DATA/kaggle-gibberish/Gibberish.xls` from
     https://www.kaggle.com/datasets/johnwdata/gibberish-text-classification
   - `DATA/hf-sowmya15/{train,validation,test}.csv` from
     https://huggingface.co/datasets/Sowmya15/gibberish_multilingual
2. Download the gibberish-detector training corpus. Its sha256 hash is
   checked by `run.py`:

   ```
   mkdir -p DATA/baselines
   curl -L -o DATA/baselines/big.txt https://raw.githubusercontent.com/domanchi/gibberish-detector/ecac96918e72fbf702cfee3274dbf280009171ad/examples/big.txt
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
   | `paper.scripts.benchmark.data --data-root DATA` | `manifest.json` (checks hashes, lists row ids) |
   | `paper.scripts.benchmark.run --data-root DATA` | predictions for A and B, `metrics.json`, `results.md`, `environment.json` |
   | `paper.scripts.benchmark.footprint --data-root DATA` | `footprint.json` |
   | `paper.scripts.benchmark.tune --data-root DATA` | `tuning/`: strategy score cache, tuning row ids, learned weights, `tuning.json`, `tuning.md` |

`run --metrics-only` recomputes the metrics and report from saved
predictions without scoring.

Scores are deterministic, so a rerun reproduces the predictions exactly.
`tune` reuses the score cache unless given `--rescore`; its Optuna search
is seeded and reproduces the same weights, and it checks that the real
`EnsembleDetector` with the learned weights reproduces every held-out
decision.
Footprint timings depend on the machine; the recorded machine is in
`environment.json` and `footprint.json`.

## Files

- `data.py`: the A and B loaders, hash checks and the manifest.
- `methods.py`: the score and decision wrappers for the four methods:
  pygarble `english` and `english_fusion`, gibberish-detector and lingua.
- `tune.py`: the weight tuning study; caches every strategy's score, learns
  a weight per strategy and a threshold on a 20 percent sample with
  Optuna, and evaluates on the held-out 80 percent (`results/tuning/`).
- `metrics.py`: default-decision metrics and the bootstrap, shared by
  `run.py` and `verify.py`.
- `report.py`: renders `results.md`.
- `figures.py`: the preprint figure, tables and number macros (`make preprint`).
- `test_benchmark.py`: one behaviour test per script.
