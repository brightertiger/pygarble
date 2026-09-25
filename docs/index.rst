pygarble: English gibberish detection
=====================================

**Deterministic, lightweight gibberish detection for English text.**

pygarble combines fixed character models, English word patterns, keyboard paths,
and encoding checks. It has no runtime dependencies, training, or downloads.
Meaningful Hindi and other non-English text may be flagged; this is expected
for English-specific scoring. It is not a language identifier or a semantic
nonsense detector. Scores are heuristics, not calibrated probabilities.

See :doc:`installation` to install the package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   cli
   calibration
   strategy-guide
   strategies
   api
   examples
   migration
   contributing

Quick Start
-----------

.. code-block:: python

   from pygarble import EnsembleDetector

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

The repository preserves its legacy benchmark separately from reviewed English
labels and a small authored challenge set. Reported engineering results are not
production precision estimates. Run ``python regression/evaluate.py --split all``
to reproduce metrics, or add ``--details`` for per-category errors.
