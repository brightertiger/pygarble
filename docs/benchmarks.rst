Benchmarks and regression checks
================================

The paper's reproducible study is the reference for reported gibberish
benchmark results. It compares fixed pygarble configurations with a pinned,
locally executed DistilBERT classifier on the complete labelled release of the
Gaskell--Bowern corpus. See the `paper and study guide
<https://github.com/brightertiger/pygarble/tree/main/paper>`_ and
`recorded results
<https://github.com/brightertiger/pygarble/tree/main/paper/study/full-results>`_.
The manuscript is not peer reviewed or published on arXiv; submission is
currently blocked by category endorsement.

What was measured
-----------------

All 109 source files were split into 79,969 consecutive chunks of at most
400 characters. The English comparison contains 173 gibberish chunks and
5,200 meaningful chunks from four English source files. The other 67 meaningful
files are a separate language diagnostic, not labelled gibberish.

.. list-table:: English comparison: observed results
   :header-rows: 1
   :widths: 35 20 20 25

   * - Configuration
     - Accuracy
     - Gibberish recall
     - False alarms / 5,200
   * - pygarble English profile
     - 96.93%
     - 28.32%
     - 41
   * - pygarble word lookup
     - 100.00%
     - 100.00%
     - 0
   * - DistilBERT, noise/word-salad policy
     - 99.87%
     - 99.42%
     - 6

Always keeping text achieves 96.78% accuracy on this imbalanced comparison
while detecting no gibberish. Word lookup's zero errors are specific to this
collection, not a universal accuracy claim. Chunks from a source are dependent,
labels were inherited without a new audit, and neural training-data overlap
is unknown. The paper reports every tested configuration, confusion counts,
source-level endpoints and language failures. Chunk summaries were added after
scoring; the frozen source-level endpoints remain available.

Median single-thread CPU scoring latency on an Apple M2 was 0.034 ms for word
lookup and 36.369 ms for DistilBERT on the same fixed 76-input timing workload.
These timings do not establish production latency, GPU performance or energy
savings. The study does not evaluate secret, PII or profanity detection.

Reproduction
------------

Run from the repository root. The study harness requires Python 3.12; the
library's supported Python versions are unchanged.

.. code-block:: bash

   # Fast, offline checks of analysis and saved target predictions; no model.
   make benchmark-check

   # Use a separate environment for the optional neural benchmark.
   python3.12 -m venv /tmp/pygarble-benchmark
   source /tmp/pygarble-benchmark/bin/activate
   python -m pip install -r paper/study/requirements-hf.txt
   make benchmark-prepare
   make benchmark

Preparation downloads checksum-pinned corpus and model assets. Full scoring
runs locally and took about 49 minutes on the measured laptop. New runs go to
ignored ``paper/study/reproduction-full/``, preserving the paper's frozen
``full-results/``. Resumption rejects changed scoring code. Follow
``paper/study/README.md`` for full artifact verification, the offline neural
subset replay and separate runtime measurements. Do not run timings alongside
other heavy workloads.

When detector code changes, evaluate it as a new version and report the code
revision and results separately. Do not overwrite paper measurements or tune
on this evaluation and describe it as an untouched test set.

Engineering checks have a different purpose
-------------------------------------------

``regression/`` retains small authored challenge sets, frozen expected outputs,
structured-text negatives and screening vectors. These detect behavioral and
compatibility regressions quickly; they are not independent research benchmarks.
CI runs these checks alongside the study's standard-library tests, without
installing or running DistilBERT.

The scanner throughput tool uses a synthetic corpus with planted findings.
It remains separate because the paper evaluates gibberish rather than scanner
accuracy or throughput. See :doc:`contributing` for the maintenance workflow.
