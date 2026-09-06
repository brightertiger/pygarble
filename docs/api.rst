API Reference
=============

Language strategies target English. Non-English text, including meaningful Hindi,
may be classified as gibberish. Scores are heuristic values, not calibrated
probabilities; the library does not establish semantic meaning or identify languages.

This reference describes the upcoming 0.9.0 API. See :doc:`installation` for
source installation and :doc:`migration` for changed behavior. Public imports
are available from ``pygarble``.

GarbleDetector
--------------

.. autoclass:: pygarble.detector.GarbleDetector
   :members: predict, predict_proba, score, analyze, applicable

EnsembleDetector
----------------

.. autoclass:: pygarble.ensemble.EnsembleDetector
   :members: predict, predict_proba, score, analyze

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

``GarbleDetector`` requires a ``Strategy`` enum member, not its string value.
``EnsembleDetector`` accepts either a named ``profile`` or a nonempty list of
``strategies``. Select one mechanism. ``weights`` correspond to strategy-list
order and are required for ``voting="weighted"``. Weights must be finite,
nonnegative, correctly sized, and not all zero.

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

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

``EnsembleDetector()`` selects the ``english`` profile. See :doc:`strategy-guide` to choose checks and :doc:`strategies` for
its current members. Profiles use union voting by default; an explicit strategies
list defaults to majority voting. Configure members independently through
``strategy_kwargs={Strategy.MARKOV_CHAIN: {"min_length": 4}}``.

Only applicable members participate. ``any`` and ``all`` use maximum and minimum
scores; ``average`` and ``weighted`` use applicable means. Majority decisions
require strictly more than half the applicable members to cross the threshold,
while the reported score is their mean. Thresholding that mean may therefore
produce a different decision. Zero-weight members abstain from weighted decisions.

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

Unknown legacy strategy options emit ``DeprecationWarning``. Scores and decisions
may change after the documented preprocessing and correctness fixes, including
when the ``legacy`` strategy set is selected.

Result records
--------------

.. autoclass:: pygarble.analysis.Analysis
   :members:

.. autoclass:: pygarble.analysis.Signal
   :members:

.. autoclass:: pygarble.analysis.Span
   :members:

Spans use offsets into the original Python string. Analysis can be converted with
``dataclasses.asdict`` and serialized as JSON. The allowlist applies only to shared
English character scoring; raw encoding/control evidence is retained.

``Analysis.status`` is ``garbled`` when the decision is positive, ``clean`` when
applicable evidence does not flag the input, or ``insufficient_evidence`` when
no member participates. ``clean`` is a detector status, not a guarantee of meaning.
``Signal.strategy`` is the strategy's string identifier. ``Signal.applicable``
indicates participation, and reasons describe heuristic evidence. Some strategies
report no spans even when their score is positive.

``Analysis.model_version`` identifies the inference contract and is distinct from
``pygarble.__version__``. Record both, along with configuration, when persisting
results. Determinism assumes fixed package, settings, and Python/Unicode tables.
