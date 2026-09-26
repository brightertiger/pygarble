Benchmarks and regression checks
================================

The paper's reproducible study is the reference for reported gibberish
benchmark results. It compares fixed pygarble configurations with a pinned,
locally executed DistilBERT classifier on the complete labelled release of the
Gaskell--Bowern corpus. See the `paper and study guide
<https://github.com/brightertiger/pygarble/tree/main/paper>`_ and
`recorded results
<https://github.com/brightertiger/pygarble/tree/main/paper/study/full-results>`_.
The manuscript, **pygarble: Modular, Low-Cost Gibberish Detection and Text
Screening in Python**, is authored by **Ujjwal Singh Rao**.
:download:`Download the paper PDF <../paper/study/manuscript.pdf>`.
The manuscript is not peer reviewed or published on arXiv; submission is
currently blocked by category endorsement.

What was measured
-----------------

All 109 source files were split into 79,969 consecutive chunks of at most
400 characters. The English comparison contains 173 gibberish chunks and
5,200 meaningful chunks from four English source files. The other 67 meaningful
files are a separate language diagnostic, not labelled gibberish.

.. list-table:: English comparison: observed metrics (%)
   :header-rows: 1

   * - Configuration
     - Accuracy
     - Precision
     - Recall
     - F1
     - Balanced accuracy
   * - English profile
     - 96.93
     - 54.44
     - 28.32
     - 37.26
     - 63.77
   * - Extended profile
     - 97.06
     - 55.73
     - 42.20
     - 48.03
     - 70.54
   * - Legacy profile
     - 97.51
     - 88.24
     - 26.01
     - 40.18
     - 62.95
   * - Word lookup
     - 100.00
     - 100.00
     - 100.00
     - 100.00
     - 100.00
   * - Entropy
     - 96.78
     - Undefined
     - 0.00
     - 0.00
     - 50.00
   * - DistilBERT: non-clean
     - 72.19
     - 10.38
     - 100.00
     - 18.80
     - 85.63
   * - DistilBERT: noise/word salad
     - 99.87
     - 96.63
     - 99.42
     - 98.01
     - 99.65
   * - Always keep
     - 96.78
     - Undefined
     - 0.00
     - 0.00
     - 50.00

.. list-table:: Confusion counts on the same 5,373 chunks
   :header-rows: 1

   * - Configuration
     - TP: caught gibberish
     - FN: missed gibberish
     - FP: false alarms
     - TN: meaningful retained
   * - English profile
     - 49
     - 124
     - 41
     - 5159
   * - Extended profile
     - 73
     - 100
     - 58
     - 5142
   * - Legacy profile
     - 45
     - 128
     - 6
     - 5194
   * - Word lookup
     - 173
     - 0
     - 0
     - 5200
   * - Entropy
     - 0
     - 173
     - 0
     - 5200
   * - DistilBERT: non-clean
     - 173
     - 0
     - 1494
     - 3706
   * - DistilBERT: noise/word salad
     - 172
     - 1
     - 6
     - 5194
   * - Always keep
     - 0
     - 173
     - 0
     - 5200

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

How to interpret the comparison
-------------------------------

The source dataset contains complete text files. The 400-character chunking
and optional source-file voting rule were introduced by this study. A file is
flagged by that secondary rule when at least half its chunks are flagged;
this aggregation is not used in the chunk confusion matrices above.

Word lookup is a separate strategy, not a member of the English profiles.
It uses lexical unfamiliarity and therefore benefits from this collection's
invented-word content. The default English profile caught 49 of 173 gibberish
chunks despite 96.93% accuracy. Its additional signals target other defects;
adding strategies does not establish superiority on every dataset.

The neural policies use the same four-class model and probabilities.
``non-clean`` flags mild gibberish, noise and word salad; ``noise/word salad``
flags only those two winning labels. The policy choice accounts for 1,488
additional false alarms on meaningful English chunks. Both policies were
fixed before inference and both are reported.

Language scope is a substantial limitation. Under the study's source-file
rule, word lookup flagged 49 of 67 meaningful non-English files and the strict
neural policy flagged 52. Word lookup can also score zero on text without
eligible Latin-letter tokens. Neither result establishes language-independent
understanding. Use :doc:`strategy-guide` to select signals for your application.

Sources
-------

* `Gaskell and Bowern's corpus paper
  <https://ceur-ws.org/Vol-3313/paper4.pdf>`_ and the
  `released corpus <https://github.com/danielgaskell/voynich>`_;
  revision ``d076a7d081f35098fa405928239595afd2e75927``.
* `Jindal's AutoNLP gibberish classifier
  <https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457>`_;
  revision ``76672dd7d3575f68ab980705bcec975cc62de71c``,
  DOI ``10.57967/hf/2664``.
* `DistilBERT <https://arxiv.org/abs/1910.01108>`_ describes the neural
  architecture. Fine-tuning data overlap with this corpus remains unknown.

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

``paper/regression/`` retains small authored challenge sets, frozen expected outputs,
structured-text negatives and screening vectors. These detect behavioral and
compatibility regressions quickly; they are not independent research benchmarks.
CI runs these checks alongside the study's standard-library tests, without
installing or running DistilBERT.

The scanner throughput tool uses a synthetic corpus with planted findings.
It remains separate because the paper evaluates gibberish rather than scanner
accuracy or throughput. See :doc:`contributing` for the maintenance workflow.
