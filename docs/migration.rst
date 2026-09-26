Upgrading
=========

Unreleased: focused modules and local backends
----------------------------------------------

The current source adds ``pygarble.screening`` and ``pygarble.gibberish``.
These changes follow 0.11.0 and do not change existing detector defaults,
scores or CLI commands. See :doc:`installation` for using an unreleased
checkout, and :doc:`architecture` for the complete module layout.

New code can import from the focused modules. Existing imports remain valid:

.. code-block:: python

   from pygarble import EnsembleDetector as ExistingDetector
   from pygarble.gibberish import EnsembleDetector
   from pygarble.pii import PIIDetector as ExistingPII
   from pygarble.screening import PIIDetector

   assert ExistingDetector is EnsembleDetector
   assert ExistingPII is PIIDetector

Old module paths such as ``pygarble.core``, ``pygarble.strategies.base``
and ``pygarble.pii.patterns`` point to the canonical implementations. Classes,
enums and caches are shared, so mixing old and new imports is supported.
Pickles referencing old paths still load. Newly created pickles use canonical
paths and are not guaranteed to load in earlier versions.

Choose the scanner deliberately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pygarble import Scanner as CombinedScanner
   from pygarble.screening import Scanner

   text = "qxzjkwpv bnmqwer zzxqv"
   assert CombinedScanner().scan(text).flagged
   assert not Scanner().scan(text).flagged

``pygarble.Scanner`` still runs all four categories by default.
``pygarble.screening.Scanner`` runs secrets, PII and profanity, and accepts
optional backends and custom detectors. It does not accept gibberish
``profile``, ``threshold`` or ``allowlist`` settings; use the gibberish API
separately, or retain the combined scanner. The standalone
``profanity_allowlist`` setting configures profanity exceptions.

Finding and redaction types are shared. Both scanners support ``scan``,
``scan_batch``, ``iter_scan`` and ``redact``. The standalone convenience
functions construct a scanner per call, while the existing top-level
functions retain their configuration cache. Reuse a scanner for repeated
work, particularly when optional backends are enabled.

CLI and optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pygarble-screen`` (also ``python -m pygarble.screening``) reads whole
documents and omits source text from scan JSON unless ``--include-text`` is
set. It supports multiline private-key redaction. Existing ``pygarble scan``
and ``pygarble redact`` keep their line-oriented behavior and existing
options; scan output still includes input text.

The base install remains dependency-free. Extras provide phonenumberslite,
python-stdnum and detect-secrets; Gitleaks is installed separately. Installing
these tools alone does not change results: select ``backends`` explicitly.
Their findings may extend native coverage or change which overlapping kind
labels a redaction. See :doc:`standalone-screening` for configuration,
failure handling and limits.

Upgrading to 0.11.0
-------------------

Version 0.11.0 has no breaking changes. It adds :class:`pygarble.Scanner`,
:func:`pygarble.scan`, :func:`pygarble.redact`, the ``secrets``, ``pii`` and
``profanity`` detectors, and the ``pygarble scan`` and ``pygarble redact``
commands. The gibberish API, its profiles, defaults and scores are unchanged.
See :doc:`screening` to get started.

Upgrading to 0.10.0
-------------------

Version 0.10.0 has no breaking changes: every 0.9.0 name, profile, keyword
argument and default is unchanged. It adds the ``pygarble`` console script
(also ``python -m pygarble``), the ``llm_output`` profile, and
``pygarble.calibrate`` for choosing a threshold from labeled samples. The new
JSON data tables under ``pygarble/data/`` ship in the repository and source
distribution only, not the wheel, and the detector does not read them at
runtime. 0.9.0 was never published to PyPI, so upgrades from 0.8.0 should also
read the 0.9.0 notes below.

Upgrading to 0.9.0
------------------

Version 0.9.0 replaces the 0.8 strategy list with profiles and the analyze API.
Existing imports from ``pygarble`` and ``pygarble.core`` remain available; no
runtime dependencies were added.

Review default decisions
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
