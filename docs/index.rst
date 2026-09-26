pygarble: a first line of defence for text
==========================================

**Local secrets, PII and profanity screening, plus independent gibberish
detection. No LLM calls or inference-model downloads.**

pygarble screens text with fixed rules, checksums, a word list, character
statistics and encoding checks. The base install has no runtime dependencies;
optional local backends extend coverage when explicitly selected. Every
finding has a kind, span, confidence and reason, without the matched text.
This documentation includes unreleased source changes; see :doc:`installation`
for installing them and :doc:`migration` for compatibility.

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   standalone-screening
   secrets
   pii
   profanity
   cli
   screening
   strategy-guide
   strategies
   calibration
   api
   examples
   migration
   architecture
   benchmarks
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

The :doc:`benchmarks` guide reports the paper's complete published-corpus
comparison with local DistilBERT and explains its limitations. Fast authored
regression fixtures remain separate from research results. Run
``make benchmark-check`` for offline study checks; the full optional neural
run is documented in the guide.
