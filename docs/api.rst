API Reference
=============

Language strategies target English. Non-English text, including meaningful Hindi,
may be classified as gibberish. Scores are heuristic values, not calibrated
probabilities; the library does not establish semantic meaning or identify languages.

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

Profiles and aggregation
------------------------

``EnsembleDetector()`` selects the ``english`` profile. See :doc:`strategies` for
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
