pygarble: a first line of defence for text
==========================================

**Deterministic, zero-dependency text screening: secrets, PII, profanity and
gibberish, with redaction.**

pygarble screens text with fixed rules, checksums, a word list, character
models and encoding checks. It has no runtime dependencies, training, or
downloads. Every finding has a kind, span, confidence and reason, and never
carries the matched text. See :doc:`screening` for the scanner and redaction.

.. code-block:: python

   from pygarble import redact, scan

   report = scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE")
   assert report.kinds() == ("aws_access_key_id", "email")
   assert redact("mail jane@example.com").text == "mail [EMAIL]"

The gibberish category is English-specific. Meaningful Hindi and other non-English text may be flagged; this is expected
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
   screening
   standalone-screening
   secrets
   pii
   profanity
   strategy-guide
   strategies
   api
   examples
   migration
   contributing

Gibberish quick start
---------------------

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
