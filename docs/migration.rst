Upgrading to 0.9.0
==================

Version 0.9.0 replaces the 0.8 strategy list with profiles and the analyze API.
Existing imports from ``pygarble`` and ``pygarble.core`` remain available; no
runtime dependencies were added.

Review default decisions
------------------------

The default ensemble adds mojibake, keyboard adjacency, and control-character
checks to Markov, likelihood ratio, and word anomaly. Inputs such as ``qwerty``
or text containing NUL can now be flagged by these extra members.

.. code-block:: python

   from pygarble import EnsembleDetector

   current = EnsembleDetector()
   former_members = EnsembleDetector(profile="legacy")
   assert current.predict("qwerty") is True
   assert former_members.profile == "legacy"

``legacy`` restores the former three-member selection, not the former package's
exact output. Shared tokenization, structured-token exclusions, and correctness
fixes apply to that profile too. Hindi scoring as gibberish remains expected.

Scores and settings
-------------------

* ``score`` aliases ``predict_proba``; neither is a calibrated probability.
* ``WORD_LOOKUP.unknown_threshold`` now affects the score. Its default 0.5 retains
  the prior mapping; nondefault settings can change results.
* Unknown legacy strategy options emit ``FutureWarning`` attributed to your
  calling code, and will become errors in a future release. Check the accepted
  names in :doc:`strategies`.
* ``EnsembleDetector`` forwards shared keyword arguments only to members that
  accept them, with one warning for any key no selected member accepts. Use
  ``strategy_kwargs`` for per-member settings.
* Use ``strategy_kwargs`` to configure a strategy's own ``threshold`` independently
  of the detector's decision threshold.
* Invalid numeric settings, including nonfinite weights, raise ``ValueError``.
* Majority voting uses a strict majority of applicable decisions; its reported
  score is their mean and need not cross the threshold with the same result.

Input handling and limits
-------------------------

Length alone no longer forces every strategy to score 1.0 through the old implicit
1,000-character token rule. Explicitly supplying the legacy ``max_string_length``
retains that classification policy. Use ``max_input_length`` to reject oversized
inputs with ``ValueError`` instead.

Batches are fully validated before processing. Invalid entries raise ``TypeError``;
worker errors and timeouts propagate. ``timeout_per_text`` controls waits for
threaded results, not a hard execution deadline. Empty or wholly inapplicable
input returns ``False``; ``analyze`` reports ``insufficient_evidence``.

Before adopting this version, compare old and new decisions on your application's
inputs and review false positives. The repository benchmark is an engineering
regression set, not a production accuracy estimate.
