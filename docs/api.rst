API Reference
=============

Use ``pygarble.screening`` for secrets, PII and profanity, and
``pygarble.gibberish`` for gibberish detection. The original ``pygarble``
exports remain supported. This reference includes unreleased source changes;
see :doc:`installation` and :doc:`migration` for version guidance.

Standalone screening
--------------------

.. autoclass:: pygarble.screening.Scanner
   :members: scan, scan_batch, iter_scan, redact
   :inherited-members:

.. autofunction:: pygarble.screening.scan

.. autofunction:: pygarble.screening.redact

The default categories are ``secrets``, ``pii`` and ``profanity``. Native
rules are enabled; optional backends run only when explicitly selected.
There is no gibberish ``profile`` or ``threshold`` setting on this scanner.
``min_confidence`` controls whether findings flag the input and are redacted;
findings below it remain in scan reports. Reuse a ``Scanner`` for repeated
calls: the convenience functions construct an instance per call.
See :doc:`standalone-screening` for options, filtering and custom detectors.

Optional backends and extension contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pygarble.screening.backends.PhoneNumbersDetector
   :members: detect

.. autoclass:: pygarble.screening.backends.StdnumDetector
   :members: detect

.. autoclass:: pygarble.screening.backends.DetectSecretsDetector
   :members: detect

.. autoclass:: pygarble.screening.backends.GitleaksDetector
   :members: detect

.. autoclass:: pygarble.screening.ScreeningDetector

.. autoexception:: pygarble.screening.BackendError

Backend constructors require their optional package or executable. Importing
these adapter classes does not load optional dependencies. Scan errors
propagate; they do not become clean reports. A custom detector declares its
``category`` and ``kinds``, and returns findings with original-text offsets.

Compatible combined scanner
---------------------------

.. autoclass:: pygarble.Scanner
   :members: scan, scan_batch, iter_scan, redact
   :inherited-members:

.. autofunction:: pygarble.scan

.. autofunction:: pygarble.redact

This scanner keeps all four categories, including gibberish, enabled by
default. Its convenience functions reuse a cached scanner per configuration.
It does not accept the standalone scanner's ``backends`` or ``detectors``
settings. See :doc:`screening` for the existing contract.

Findings
--------

.. autoclass:: pygarble.findings.Finding

.. autoclass:: pygarble.findings.ScanReport
   :members: by_category, kinds, to_dict

.. autoclass:: pygarble.findings.Redaction

Native screening detectors
--------------------------

.. autoclass:: pygarble.screening.SecretsDetector
   :members: detect

.. autoclass:: pygarble.screening.PIIDetector
   :members: detect

.. autoclass:: pygarble.screening.ProfanityDetector
   :members: detect

Gibberish detection
-------------------

Language strategies target English. Meaningful non-English text may be
flagged; these checks do not establish semantic meaning or identify languages.
Scores are heuristic values, not calibrated probabilities.

.. autoclass:: pygarble.gibberish.Strategy

GarbleDetector
~~~~~~~~~~~~~~

.. autoclass:: pygarble.gibberish.GarbleDetector
   :members: predict, predict_proba, score, analyze, applicable

EnsembleDetector
~~~~~~~~~~~~~~~~

.. autoclass:: pygarble.gibberish.EnsembleDetector
   :members: predict, predict_proba, score, analyze

Under ``voting='majority'`` the decision counts member votes, so
``Analysis.garbled`` can be ``True`` while ``Analysis.score`` is below
threshold.

Both classes accept a string or list of strings. ``predict`` returns bools,
``score`` and ``predict_proba`` return floats, and ``analyze`` returns immutable
analysis records. Batch inputs are validated before any member is evaluated.

Configuration
-------------

Both constructors accept these settings:

.. list-table:: Common settings
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Default
     - Meaning
   * - ``threshold``
     - ``0.5``
     - Decision cutoff in [0, 1]; reaching the cutoff flags applicable evidence.
   * - ``allowlist``
     - ``None``
     - Iterable of vocabulary words for shared English scoring, normalized for
       case and diacritics. A plain string is not a valid allowlist.
   * - ``threads``
     - ``None``
     - Optional positive worker count for batches. Serial execution is the default.
   * - ``max_input_length``
     - ``None``
     - Optional positive maximum length in Python string characters per input.
   * - ``timeout_per_text``
     - ``None``
     - Optional finite positive timeout in seconds for threaded-result waits.
   * - ``strategy_kwargs``
     - ``None``
     - Settings for one strategy in ``GarbleDetector``; mapping from selected
       ``Strategy`` members to their settings in ``EnsembleDetector``.

``GarbleDetector`` accepts a ``Strategy`` enum member or its string value,
such as ``"markov_chain"``; an unknown name raises ``ValueError``.
``EnsembleDetector`` accepts either a named ``profile`` or a nonempty list of
``strategies``. Select one mechanism. ``weights`` correspond to strategy-list
order and are required for ``voting="weighted"``. Weights must be finite,
nonnegative, correctly sized, and not all zero. ``weights`` are only used with
``voting="weighted"``; passing them with any other voting mode emits a
``FutureWarning`` and they are ignored. This will become an error in a future
release.

.. code-block:: python

   from pygarble.gibberish import GarbleDetector, Strategy

   detector = GarbleDetector(
       Strategy.CONTROL_CHARACTERS,
       threshold=0.5,
       strategy_kwargs={"max_combining_run": 8},
   )
   assert detector.predict("hello\x00world") is True

Return values
-------------

.. list-table:: Scalar and batch results
   :header-rows: 1

   * - Method
     - String input
     - List of strings
   * - ``predict``
     - ``bool``
     - ``List[bool]``
   * - ``score`` / ``predict_proba``
     - ``float`` in [0, 1]
     - ``List[float]``
   * - ``analyze``
     - ``Analysis``
     - ``List[Analysis]``

Batch order is preserved, and an empty batch returns an empty list. Bytes,
generators, tuples, and nonstring batch members are not accepted by these methods.
``GarbleDetector.applicable(text)`` accepts one string and reports whether its
strategy supplies evidence; applicability does not mean the input is garbled.

Profiles and aggregation
------------------------

``EnsembleDetector()`` selects the ``english`` profile.

.. list-table:: Named profiles
   :header-rows: 1
   :widths: 25 75

   * - Profile
     - Intended use
   * - ``english``
     - Default. General English screening: Markov, likelihood ratio, word
       anomaly, mojibake, keyboard adjacency and control characters.
   * - ``english_extended``
     - Adds pattern matching, localized anomalies and repetition; more
       aggressive, with more potential false positives.
   * - ``legacy``
     - The former three-member set: Markov, likelihood ratio and word anomaly.
   * - ``corruption``
     - Mojibake and control artifacts, independent of English plausibility.
   * - ``spoofing``
     - Unicode script and confusable heuristic.
   * - ``llm_output``
     - Repetition, control characters, mojibake and local anomaly; a
       deterministic pre-check for degenerate model output that stays quiet
       on code and technical prose.
   * - ``english_fusion``
     - Opt-in. Word lookup, likelihood ratio and cross parsing combined with
       ``fisher`` voting; English plausibility only, so other languages
       written in Latin letters may be flagged.

See :doc:`strategy-guide` to choose checks and :doc:`strategies` for
its current members. Profiles use union voting by default (``english_fusion``
uses ``fisher``); an explicit strategies list defaults to majority voting. An
explicit ``voting`` argument overrides either default. Configure members independently through
``strategy_kwargs={Strategy.MARKOV_CHAIN: {"min_length": 4}}``.

Only applicable members participate. ``any`` and ``all`` use maximum and minimum
scores; ``average`` and ``weighted`` use applicable means. Majority decisions
require strictly more than half the applicable members to cross the threshold,
while the reported score is their mean. Thresholding that mean may therefore
produce a different decision. Zero-weight members abstain from weighted decisions.

``fisher`` reads each applicable member's score against a table of that
strategy's scores on a synthetic English null (word salads drawn by word
frequency) and takes the smallest tail probability, from 0.5 down to 0.001,
whose null threshold the score strictly exceeds; a score that exceeds none
gives 1. Fisher's method combines these p-values, and the reported score is
``1 - p ** (ln 0.5 / ln fisher_alpha)``, so with the default threshold of 0.5
the text is flagged when the combined p-value is at most ``fisher_alpha``
(keyword-only, default 0.001, strictly between 0 and 1). Passing
``fisher_alpha`` with another voting mode emits a ``FutureWarning`` and it is
ignored; ``weights`` are ignored under ``fisher`` as under every mode except
``weighted``. The value is a heuristic: members are correlated and the null is
synthetic, so ``fisher_alpha`` is not a guaranteed false-positive rate. The
null tables describe default-constructed strategies, so member settings and
allowlists shift scores without shifting the tables. The tables load only
when ``fisher`` voting is used.

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector

   fusion = EnsembleDetector(profile="english_fusion")
   assert fusion.voting == "fisher"
   assert fusion.predict("The committee will meet again next week.") is False

An empty input or a set with no applicable members yields ``False`` and
``insufficient_evidence``. This does not certify meaningful English.

Limits and compatibility
------------------------

``max_input_length`` raises ``ValueError`` for oversized scalar or batch input.
The legacy opt-in ``max_string_length`` still classifies long non-URL tokens as
suspicious. There is no longer a universal implicit long-string decision.

``threads`` must be a positive integer. ``timeout_per_text`` affects waits for
threaded results, not scalar/serial execution or a hard wall-clock deadline.
Python workers cannot be killed; executor shutdown may wait. Errors and timeouts
propagate rather than producing clean fallback predictions.

Unknown legacy strategy options emit ``FutureWarning``, attributed to the line
in your code that constructed the detector; they will become errors in a future
release. ``EnsembleDetector`` forwards its shared keyword arguments only to the
member strategies that accept them, and warns once for any key that no selected
member accepts. Unknown keys in ``strategy_kwargs`` warn once per member and are
not passed on. Scores and decisions may change after the documented preprocessing and correctness fixes, including
when the ``legacy`` strategy set is selected.

Calibration
-----------

.. autofunction:: pygarble.gibberish.calibrate

.. autoclass:: pygarble.gibberish.CalibrationReport

.. autoclass:: pygarble.gibberish.ThresholdPoint

``calibrate(detector, garbled, clean, *, objective="f1",
max_false_positive_rate=None, thresholds=None)`` accepts any detector with a
``score`` method that takes a list of strings. ``garbled`` and ``clean`` must
each contain at least one string. ``objective="max_fpr"`` requires
``max_false_positive_rate`` in [0, 1], and passing it with
``objective="f1"`` raises ``ValueError``.

``CalibrationReport`` is a frozen dataclass with ``recommended`` (a
``ThresholdPoint``), ``objective``, ``max_false_positive_rate``, the
``garbled`` and ``clean`` sample counts, and ``points``, a tuple of
``ThresholdPoint`` for every candidate in ascending order. ``ThresholdPoint``
holds ``threshold``, ``precision``, ``recall``, ``f1`` and
``false_positive_rate``. See :doc:`calibration` for a walkthrough.

Result records
--------------

.. autoclass:: pygarble.gibberish.Analysis
   :members:

.. autoclass:: pygarble.gibberish.Signal
   :members:

.. autoclass:: pygarble.gibberish.Span
   :members:

Spans use offsets into the original Python string. Analysis can be converted with
``dataclasses.asdict`` and serialized as JSON. The allowlist applies to every
strategy: allowlisted words are excluded from English scoring and blanked out of
the text that raw-text checks scan. ``SYMBOL_RATIO`` reads the raw text, because
allowlisted letters can only lower its score. Control characters and encoding
damage outside allowlisted words are still reported.

``Analysis.status`` is ``garbled`` when the decision is positive, ``clean`` when
applicable evidence does not flag the input, or ``insufficient_evidence`` when
no member participates. ``clean`` is a detector status, not a guarantee of meaning.
``Signal.strategy`` is the strategy's string identifier. ``Signal.applicable``
indicates participation, and reasons describe heuristic evidence. Some strategies
report no spans even when their score is positive.

``Analysis.model_version`` identifies the inference contract and is distinct from
``pygarble.__version__``. Record both, along with configuration, when persisting
results. Determinism assumes fixed package, settings, and Python/Unicode tables.
