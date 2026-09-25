Quick start
===========

Install pygarble using :doc:`installation` before running these
examples. Detectors operate on Python strings; decode bytes before calling them.
No training or ``fit()`` step is needed.

Screen English text
-------------------

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()
   assert detector.predict("Hello world") is False
   assert detector.predict("asdfghjkl") is True
   assert detector.predict("नमस्ते दुनिया") is True

Hindi being flagged is expected: the default checks target English. A negative
result does not establish meaning, correct grammar, or language identity.

Process a batch
---------------

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()
   texts = ["Hello world", "qxzjkwpv"]
   assert detector.predict(texts) == [False, True]
   scores = detector.score(texts)
   assert scores == detector.predict_proba(texts)
   assert all(0.0 <= score <= 1.0 for score in scores)

A list input returns a list in the same order; a string returns a single result.
Scores are heuristics, not probabilities. Start with serial execution for short
texts and measure before enabling threads.

Choose what to detect
---------------------

.. code-block:: python

   from pygarble import EnsembleDetector, GarbleDetector, Strategy

   corruption = EnsembleDetector(profile="corruption")
   assert corruption.predict("नमस्ते दुनिया") is False
   assert corruption.predict("hello\x00world") is True

   local = GarbleDetector(Strategy.LOCAL_ANOMALY)
   assert local.predict("Please review qxzjkwpvm before delivery.") is True

   keyboard = GarbleDetector(
       Strategy.KEYBOARD_ADJACENCY, keyboard_layout="azerty"
   )
   assert keyboard.predict("azerty") is True

Use ``english_extended`` to add local anomalies, repetition, and pattern matching
to the default profile. It can flag more valid text. See :doc:`strategy-guide`
for choosing checks and :doc:`strategies` for the complete settings catalog.

Use the command line
--------------------

The ``pygarble`` console script checks one text per line from files or stdin
and exits with status 1 when any line is flagged, so it drops into shell
pipelines. See :doc:`cli` for scores, JSON output and threshold calibration.

.. code-block:: console

   $ printf 'hello world\nasdfghjkl\n' | pygarble check
   clean	hello world
   garbled	asdfghjkl

Inspect a decision
------------------

.. code-block:: python

   from pygarble import EnsembleDetector

   result = EnsembleDetector().analyze("hello\x00world")
   assert result.garbled is True
   assert result.status == "garbled"
   for signal in result.signals:
       print(signal.strategy, signal.score, signal.applicable, signal.reason)

   empty = EnsembleDetector().analyze("")
   assert empty.garbled is False
   assert empty.status == "insufficient_evidence"

Required fields need a separate empty-input check. For JSON output and spans,
see :doc:`examples`. For thresholds, voting, limits, and errors, see :doc:`api`.
