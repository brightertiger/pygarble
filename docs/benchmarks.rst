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
  It flags 33.2 percent of tweets as shipped. Two causes overlap. Its
  mojibake and control-character members flag 28.5 percent of tweets,
  almost all through the control codes of mis-decoded text, which the
  dataset labels clean. Its Markov member alone flags 17.8 percent, so
  removing the two encoding checks still leaves 18.2 percent flagged.

:doc:`strategy-guide` turns these results into advice.

Tuning to a use case from a small labelled sample
-------------------------------------------------

The two profiles are fixed member lists. With a labelled sample of your own
data you can instead learn how much each strategy should count. The tuning study
learns a weight from 0 to 1 for each of the 32 strategies and a threshold
from 0.05 to 0.80, under ``weighted`` voting. A text's score is the weighted
mean of the scores of the members that apply to it. A weight of 0 switches a
strategy off.

The tuning sample is a stratified 20 percent of each dataset. Rows with
identical text are kept on the same side of the split, so no tweet in the
held-out set also appears in the tuning sample. Optuna's TPE
sampler searches 1,000 configurations for the best F1 on that sample, with
junk as the positive class. The search was still improving when it
stopped: in every draw the best configuration came after trial 832. Each
strategy is scored once per row and cached,
so the search takes about 17 seconds per sample on one core. The study
repeats this for five seeds. Each result is rebuilt as an
``EnsembleDetector`` and evaluated on the held-out 80 percent. Both profiles
are scored on the same rows. The study was added after the main results and
uses labelled rows from both datasets, so it is post hoc in that sense, but
no evaluation row is seen during tuning.

.. list-table:: Held-out 80 percent, primary draw (seed 20261011): 2,810 Survey Junk rows, 3,200 random strings, 4,777 tweets
   :header-rows: 1

   * - Method
     - Survey Junk capture
     - Precision
     - Recall
     - FPR
   * - ``english`` (default)
     - 0.826 (0.811 to 0.840)
     - 0.653 (0.639 to 0.666)
     - 0.949 (0.941 to 0.956)
     - 0.338 (0.324 to 0.351)
   * - ``english_fusion``
     - 0.891 (0.879 to 0.903)
     - 0.904 (0.894 to 0.913)
     - 0.990 (0.987 to 0.994)
     - 0.071 (0.064 to 0.078)
   * - tuned weights
     - 0.868 (0.856 to 0.882)
     - 0.941 (0.934 to 0.949)
     - 0.967 (0.960 to 0.973)
     - 0.040 (0.035 to 0.046)

Intervals are 95 percent bootstrap intervals, in parentheses.

.. list-table:: Held-out 80 percent, mean ± standard deviation over five draws
   :header-rows: 1

   * - Method
     - Survey Junk capture
     - Precision
     - Recall
     - FPR
   * - ``english`` (default)
     - 0.830 ± 0.005
     - 0.653 ± 0.002
     - 0.946 ± 0.002
     - 0.337 ± 0.003
   * - ``english_fusion``
     - 0.896 ± 0.004
     - 0.906 ± 0.002
     - 0.991 ± 0.001
     - 0.069 ± 0.002
   * - tuned weights
     - 0.897 ± 0.027
     - 0.930 ± 0.016
     - 0.976 ± 0.023
     - 0.049 ± 0.012

The tuned weights flag fewer tweets than ``english_fusion`` and have higher
precision in every draw, at about the same capture on average. They trade
away some recall: it is lower on average and in four of the five draws, and
on the primary draw both capture and recall are below ``english_fusion``'s.
The tuned rows vary more between draws because a 20 percent sample is
small. The largest mean weights went to ``word_lookup`` (1.00, full weight
in every draw), ``zipf_conformity`` (0.81), ``entropy_based`` (0.78) and
``hex_string`` (0.76); below ``word_lookup`` the ranking moves between
draws. ``control_characters`` and ``mojibake`` have mean weights of 0.04
and 0.03. In the primary draw ``control_characters``, ``pattern_matching``
and ``mojibake`` have weight 0, so the search gave almost no weight to the
encoding-damage checks that this dataset's labels penalise. Thresholds
ranged from 0.13 to 0.16. The learned weights describe these two datasets,
not junk in general.

The primary draw's configuration, built directly:

.. code-block:: python

   from pygarble import EnsembleDetector

   # Learned on the preprint's primary tuning sample (seed 20261011).
   weights = {
       "local_anomaly": 0.05,
       "entropy_based": 1.0,
       "vowel_ratio": 0.05,
       "keyboard_pattern": 0.35,
       "markov_chain": 0.8,
       "ngram_frequency": 0.65,
       "word_lookup": 1.0,
       "symbol_ratio": 0.85,
       "repetition": 0.3,
       "hex_string": 0.85,
       "pronounceability": 0.65,
       "unicode_script": 0.45,
       "bigram_probability": 0.55,
       "letter_position": 0.25,
       "consonant_sequence": 0.5,
       "vowel_pattern": 0.65,
       "letter_frequency": 0.75,
       "rare_trigram": 0.7,
       "function_word_density": 0.45,
       "affix_detection": 0.15,
       "zipf_conformity": 0.25,
       "word_collocation": 0.05,
       "log_likelihood_ratio": 0.9,
       "word_anomaly": 1.0,
       "keyboard_adjacency": 0.6,
       "cross_parsing": 0.6,
       "primed_compression": 0.95,
       "ngram_rank": 0.05,
       "permutation_test": 0.25,
   }
   tuned = EnsembleDetector(
       strategies=list(weights),
       weights=list(weights.values()),
       voting="weighted",
       threshold=0.16,
   )
   assert tuned.predict("asdfghjkl qwpoeiru")
   assert not tuned.predict("Thanks for the quick reply, see you on Monday!")

To tune for your own data, score a labelled sample with each strategy and
search the weights and threshold. The benchmark's ``tune.py`` does this with
Optuna; all learned weights are in ``paper/benchmark/results/tuning/``.


Reproduction
------------

Per-row scores and decisions for every method are committed. The raw data is
never committed. From the repository root:

.. code-block:: bash

   # Recompute every metric from the saved predictions, and the tuning
   # results from the cached strategy scores; needs numpy only.
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

``paper/regression/`` keeps a golden corpus of frozen expected outputs, whose
inputs are a 600-row sample of the preprint benchmark's open dataset plus
hand-written edge cases, and structured-text negatives and screening vectors. They catch
behaviour and compatibility regressions quickly. They are not research
benchmarks. CI runs them alongside ``make benchmark-verify``.

The scanner throughput tool uses a synthetic corpus with planted findings.
It stays separate because the preprint measures gibberish, not scanner
accuracy or throughput. See :doc:`contributing` for the maintenance workflow.
