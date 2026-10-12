pygarble: a first line of defence for text
==========================================

**Local secrets, PII and profanity screening, plus independent gibberish
detection. No LLM calls or inference-model downloads.**

pygarble screens text with fixed rules, checksums, a word list, character
statistics and encoding checks. The base install has no runtime dependencies;
optional local backends extend coverage when explicitly selected. Every
finding has a kind, span, confidence and reason, without the matched text.
This documentation describes 0.11.0; see :doc:`installation` for installing it
and :doc:`migration` when upgrading from 0.8.0.

.. code-block:: python

   from pygarble.screening import Scanner

   scanner = Scanner(max_input_length=100_000)
   report = scanner.scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE")
   assert report.kinds() == ("aws_access_key_id", "email")
   assert scanner.redact("mail jane@example.com").text == "mail [EMAIL]"

Use :doc:`standalone-screening` for the three rule categories, optional
backends and document CLI. Use ``pygarble.gibberish`` for gibberish detection.
The original ``from pygarble import Scanner`` still runs all four categories;
its guide is :doc:`screening`. See :doc:`architecture` for module ownership
and old import pointers.

The gibberish category is English-specific. Meaningful Hindi and other non-English text may be flagged; this is expected
for English-specific scoring. It is not a language identifier or a semantic
nonsense detector. Scores are heuristics, not calibrated probabilities.

See :doc:`installation` to install the package.

Research and measured results
-----------------------------

A preprint measures the gibberish detector on two public English datasets,
Survey Junk and Tweets vs. Random Strings, against two CPU baselines at
default settings. The ``english_fusion`` profile captures 89.5% of the survey
junk and recalls 99.1% of the random strings at a precision of 0.906, from a
1.05 MB install. A Markov-chain baseline flags fewer tweets, a language
identifier captures more survey junk, and the default ``english`` profile
flags 33.2% of tweets. Read :doc:`benchmarks` for every result, the
limitations and the reproduction commands.

Citing pygarble
---------------

Cite the preprint for the benchmark results and the software entry for the
library. The preprint is archived on Zenodo under DOI
`10.5281/zenodo.23312785 <https://doi.org/10.5281/zenodo.23312785>`_, which always resolves to the latest version.

.. code-block:: bibtex

   @misc{rao2026pygarble,
     author       = {Rao, Ujjwal Singh and Singh, Ricky},
     title        = {How Much Junk Do Rules Catch? Evaluating and Tuning pygarble, a Dependency-Free Gibberish Screener, on Survey Responses and Social-Media Text},
     year         = {2026},
     howpublished = {Preprint, Zenodo},
     doi          = {10.5281/zenodo.23312785},
     url          = {https://doi.org/10.5281/zenodo.23312785}
   }

   @misc{pygarble,
     author       = {Rao, Ujjwal Singh and Singh, Ricky},
     title        = {pygarble: local screening of text for secrets, {PII}, profanity and gibberish},
     year         = {2026},
     howpublished = {\url{https://github.com/brightertiger/pygarble}},
     note         = {Version 0.11.0, https://github.com/brightertiger/pygarble/releases/tag/v0.11.0}
   }

.. The captions below become the navigation tabs of the Material theme.

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 2
   :caption: Screening

   installation
   quickstart
   standalone-screening
   secrets
   pii
   profanity
   cli
   screening
   examples
   migration

.. toctree::
   :maxdepth: 2
   :caption: Gibberish

   strategy-guide
   strategies
   calibration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   architecture
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Maintainers

   contributing
   publishing

Gibberish quick start
---------------------

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector

   detector = EnsembleDetector()
   detector.predict("Hello world")     # False
   detector.predict("asdfghjkl")       # True
   detector.predict("नमस्ते दुनिया")    # True: English-specific checks
   detector.predict("hello\x00world")  # True: control artifact

Use ``profile="corruption"`` to check encoding/control artifacts independently
of English plausibility, or ``profile="english_extended"`` for more aggressive
localized and repetition detection. See :doc:`api` for explanations, batching,
per-strategy configuration, and abstention behavior.

Evaluation
----------

The :doc:`benchmarks` guide reports the preprint benchmark and explains its
limitations. It uses two public English datasets: *Survey Junk* (Kaggle, 3,513
junk-only rows, measuring the capture rate of junk) and *Tweets vs. Random
Strings* (Hugging Face, 9,971 rows, measuring precision, recall and
false-positive rate). The baselines, gibberish-detector 0.1.1 and lingua
2.2.0, were run as installed with default settings, so the comparison measures
out-of-the-box behaviour. Fast engineering regression checks remain separate
from research results. Run ``make benchmark-verify`` to recompute the preprint
metrics from the committed predictions (see ``paper/scripts/benchmark/README.md``
to re-score from raw data); the full write-up is at
`doi:10.5281/zenodo.23312785 <https://doi.org/10.5281/zenodo.23312785>`_.
