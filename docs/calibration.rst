Calibrating the threshold
=========================

Scores are heuristic, not probabilities, so the right decision threshold
depends on your data. ``calibrate`` scores labeled samples once and sweeps
every observed score as a candidate threshold.

.. code-block:: python

   from pygarble import EnsembleDetector, calibrate

   garbled = ["qxzjkwpv bnmqwer", "asdfghjkl"]
   clean = ["hello world", "please send the invoice"]
   report = calibrate(EnsembleDetector(), garbled, clean)
   detector = EnsembleDetector(threshold=report.recommended.threshold)
   assert detector.predict(garbled) == [True, True]
   assert detector.predict(clean) == [False, False]

``objective="f1"`` (default) maximises F1. ``objective="max_fpr"`` with
``max_false_positive_rate=0.01`` picks the highest recall whose
false-positive rate stays at or below one percent, or falls back to the
``1.0`` cut if no threshold qualifies. Ties resolve to the higher observed
cut, and the recommended threshold is the midpoint of the gap below that
cut, so it keeps a margin on both sides.

.. code-block:: python

   from pygarble import EnsembleDetector, calibrate

   report = calibrate(
       EnsembleDetector(),
       ["asdfghjkl", "qxzjkwpv"],
       ["hello world", "please send the invoice"],
       objective="max_fpr",
       max_false_positive_rate=0.0,
   )
   assert report.recommended.false_positive_rate == 0.0
   assert report.objective == "max_fpr"

``report.points`` lists precision, recall, F1 and false-positive rate at
every candidate. Pass ``thresholds=[...]`` to sweep your own candidates
instead; the recommended threshold is then one of them exactly. Under
``voting="majority"`` the ensemble counts member votes, so the threshold
applies per member; the report still measures the aggregate score.

The same sweep is available from the shell as ``pygarble calibrate``; see
:doc:`cli`.
