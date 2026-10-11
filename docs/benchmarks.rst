Benchmark and preprint
======================

The preprint *How Much Junk Do Rules Catch?* measures pygarble's gibberish
detector on two public English datasets. It compares pygarble with two CPU
baselines, with every method at its own default decision. Read the
`preprint PDF <https://github.com/brightertiger/pygarble/blob/main/paper/preprint/main.pdf>`_
or its `LaTeX source
<https://github.com/brightertiger/pygarble/blob/main/paper/preprint/main.tex>`_.
The `frozen protocol
<https://github.com/brightertiger/pygarble/blob/main/paper/benchmark/protocol.md>`_
and its `dated deviations
<https://github.com/brightertiger/pygarble/blob/main/paper/benchmark/deviations.md>`_
define what was done. Every number on this page comes from the committed
results and can be recomputed.

What was measured
-----------------

**Survey Junk.** The Kaggle dataset `Gibberish Text Classification
<https://www.kaggle.com/datasets/johnwdata/gibberish-text-classification>`_
holds free-text answers from poor-quality survey respondents. Its licence is
listed as unknown, so the repository stores row ids and hashes, never the
text. The file has 3,767 rows, all labelled junk. After stripping each answer
and dropping empty rows and exact duplicates, 3,513 rows remain. Most are
short keyboard mashes. The file has no clean rows, so it measures only the
capture rate: the share of rows a method flags. It says nothing about false
positives.

**Tweets vs. Random Strings.** The Hugging Face dataset
`gibberish_multilingual
<https://huggingface.co/datasets/Sowmya15/gibberish_multilingual>`_
is licensed Apache-2.0. The benchmark keeps its 9,971 non-empty English rows
from all three splits: 4,000 random strings labelled junk and 5,971 tweets
labelled clean. Labels are used as given, and 215 duplicate rows are kept. It
measures precision, recall and the false-positive rate (FPR), the share of
tweets flagged. Its only junk is random lower-case strings, which are easier
than real junk. 26.4 percent of the tweets carry mojibake from emoji decoded
with the wrong encoding, and the dataset labels them clean.

Both datasets are English and short text. Long documents are not measured.
PII, secret and profanity detection are not part of this benchmark.

Methods
-------

Each method runs once, at its default decision. No threshold is fitted on
either dataset.

* **pygarble** ``english``: ``EnsembleDetector()``, the default profile.
  It flags a text when any member flags it: Markov chain, log-likelihood
  ratio, word anomaly, mojibake, keyboard adjacency and control characters.
* **pygarble** ``english_fusion``: ``EnsembleDetector(profile="english_fusion")``.
  It combines word lookup, log-likelihood ratio and cross-parsing with Fisher
  voting. See :doc:`strategy-guide`.
* **gibberish-detector 0.1.1**: a character Markov-chain detector. It ships
  no model, so one is trained on the project's own ``big.txt``. A text is
  flagged when its score is above the command-line default of 4.0.
* **lingua 2.2.0**: a language identifier, with the policy that any text not
  detected as English is junk.
* **pygarble** ``english``, **tuned** (post hoc): the ``english`` members
  without mojibake and control characters, with the same voting and
  threshold. It was built after seeing the tweet results, so it is not part
  of the pre-registered comparison. The package is unchanged.

The 95 percent intervals come from 1,000 bootstrap resamples of the rows.

Results
-------

.. list-table:: Survey Junk: capture rate (3,513 junk rows)
   :header-rows: 1

   * - Method
     - Capture rate
     - 95% interval
   * - ``english`` (default)
     - 0.830
     - 0.817 to 0.843
   * - ``english_fusion``
     - 0.895
     - 0.884 to 0.905
   * - gibberish-detector
     - 0.722
     - 0.707 to 0.736
   * - lingua
     - 0.989
     - 0.985 to 0.992
   * - ``english``, tuned (post hoc)
     - 0.830
     - 0.817 to 0.843

.. list-table:: Tweets vs. Random Strings: 4,000 random strings and 5,971 tweets
   :header-rows: 1

   * - Method
     - Precision
     - Recall
     - FPR
   * - ``english`` (default)
     - 0.656 (0.645 to 0.670)
     - 0.947 (0.940 to 0.954)
     - 0.332 (0.320 to 0.344)
   * - ``english_fusion``
     - 0.906 (0.898 to 0.915)
     - 0.991 (0.987 to 0.993)
     - 0.068 (0.062 to 0.075)
   * - gibberish-detector
     - 0.995 (0.993 to 0.997)
     - 0.822 (0.809 to 0.833)
     - 0.003 (0.002 to 0.004)
   * - lingua
     - 0.876 (0.867 to 0.886)
     - 0.994 (0.991 to 0.996)
     - 0.094 (0.087 to 0.101)
   * - ``english``, tuned (post hoc)
     - 0.777 (0.766 to 0.789)
     - 0.947 (0.940 to 0.954)
     - 0.182 (0.173 to 0.192)

Intervals are in parentheses.

.. list-table:: Footprint
   :header-rows: 1

   * - Method
     - ms per 1,000 texts
     - Install (MB)
   * - ``english`` (default)
     - 66.4
     - 1.05
   * - ``english_fusion``
     - 62.0
     - 1.05
   * - gibberish-detector
     - 6.8
     - 6.51
   * - lingua
     - 1039.0
     - 307.47
   * - ``english``, tuned (post hoc)
     - 52.7
     - 1.05

Time is the default decision over all 9,971 Tweets vs. Random Strings texts
on one thread of an Apple M2 with Python 3.12.2, the median of five runs
after a warm-up. Timings depend on the machine. Install size counts the
installed files and runtime dependencies. For gibberish-detector it includes
the training corpus. lingua bundles its language models in its wheel. All
methods run offline once installed, and pygarble has no runtime dependencies.

What this shows
---------------

At default settings, ``english_fusion`` captures about nine in ten survey
junk answers and recalls nearly every random string. It flags about one
tweet in fifteen, from a one-megabyte install. Neither baseline matches it on
both datasets at once.

pygarble is beaten on single measures:

* **The Markov baseline flags far fewer tweets.** Its FPR is 0.3 percent
  against 6.8 percent for ``english_fusion``. It misses more than a quarter
  of the survey junk. When wrongly flagging a real post costs much more than
  missing junk, it is the better default.
* **lingua captures more survey junk**: 98.9 percent against 89.5 percent.
  It flags more tweets and needs a 307 MB install. It answers which language
  a text is in, not whether it is junk.
* **The default** ``english`` **profile is not suited to informal text.**
  It flags 33.2 percent of tweets as shipped. The post hoc tuned variant
  lowers that to 18.2 percent with no loss of capture or recall. The
  preprint attributes the rest to its Markov member, which alone flags many
  informal posts.

The tuned variant shows the general mechanism: a profile is a list of
strategies, and a domain that does not count some kind of damage as junk
removes the strategies that target it. :doc:`strategy-guide` turns these
results into advice.

Reproduction
------------

Per-row scores and decisions for every method are committed. The raw data is
never committed. From the repository root:

.. code-block:: bash

   # Recompute every metric from the saved predictions; needs numpy only.
   make benchmark-verify PYTHON=/path/to/venv/bin/python

   # Regenerate the figure, tables and number macros, then build the PDF.
   # Needs matplotlib and tectonic.
   make preprint PYTHON=/path/to/venv/bin/python

The first command fails on any difference from the committed metrics. CI
runs it on pull requests and pushes to main. To re-score from the raw files,
follow the
`benchmark README
<https://github.com/brightertiger/pygarble/blob/main/paper/scripts/benchmark/README.md>`_.
Scores are deterministic, so a rerun reproduces the predictions exactly.

When detector code changes, report it as a new version with its code
revision. Do not overwrite the committed predictions or results.

Engineering checks have a different purpose
-------------------------------------------

``paper/regression/`` keeps small authored challenge sets, frozen expected
outputs, structured-text negatives and screening vectors. They catch
behaviour and compatibility regressions quickly. They are not research
benchmarks. CI runs them alongside ``make benchmark-verify``.

The scanner throughput tool uses a synthetic corpus with planted findings.
It stays separate because the preprint measures gibberish, not scanner
accuracy or throughput. See :doc:`contributing` for the maintenance workflow.
