Upgrading
=========

Upgrading from 0.8.0 to 0.11.0
------------------------------

0.11.0 is the first release after 0.8.0. Versions 0.9.0 and 0.10.0 were
development milestones and were never published to PyPI, so everything below
reaches 0.8.0 users at once. Until 0.11.0 is on PyPI, ``pip install pygarble``
installs 0.8.0; see :doc:`installation` to install from source.

**0.11.0 is a breaking release relative to 0.8.0.** Imports, constructor
signatures and ``Strategy`` members from 0.8.0 keep working, but gibberish
scores and decisions change, and no setting reproduces 0.8.0 output exactly.
The `changelog <https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md>`_
lists every change.

Find what affects you
~~~~~~~~~~~~~~~~~~~~~

1. Record 0.8.0 decisions and scores on a sample of your real inputs, upgrade
   in a separate environment, and compare. Review both new flags and lost
   flags; the repository benchmark is an engineering regression set, not a
   production accuracy estimate.
2. Run your tests with deprecation and future warnings as errors. This finds
   the 0.8.0 import paths and the settings that will become errors:

   .. code-block:: bash

      python -W error::DeprecationWarning -W error::FutureWarning -m pytest

3. Check the sections below for each exception type your code catches, and for
   any code that compares scores exactly or iterates over all strategies.

Scores changed in every 0.8.0 strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strategies share one tokeniser. A whitespace-separated chunk that contains a
digit, ``://``, ``@`` or a backslash, starts with ``www.``, ``data:`` or a path
prefix, or has a lowercase-to-uppercase step (camelCase) is treated as
structured, and its letters are not scored as words. Most JWTs, base64 blobs,
GitHub-style tokens, PEM private key blocks, URLs, email addresses, paths and
code identifiers are therefore no longer flagged as gibberish. Affected
strategies include ``MARKOV_CHAIN``, ``LOG_LIKELIHOOD_RATIO``,
``NGRAM_FREQUENCY``, ``WORD_ANOMALY``, ``KEYBOARD_ADJACENCY``,
``KEYBOARD_PATTERN`` and ``LETTER_FREQUENCY``. Some inputs move the other way:
zero-width-joined letters and homoglyph domains can now be flagged. Correctness
fixes in other strategies, listed in the changelog, change further scores.

If you relied on the gibberish detector to catch credentials, use the secrets
and PII detectors instead; see :doc:`screening`.

.. code-block:: python

   from pygarble.screening import Scanner

   token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
   assert Scanner(categories=["secrets"]).scan(token).flagged

``LOG_LIKELIHOOD_RATIO``, and every ensemble containing it, also differs from
0.8.0 in the last one or two floating-point digits. Regenerate exact-equality
golden files. There is no escape hatch for these changes.

The default ensemble has six members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EnsembleDetector()`` adds mojibake, keyboard adjacency and control-character
checks to the 0.8.0 members (Markov chain, likelihood ratio and word anomaly),
still with ``any`` voting. Inputs such as ``qwerty``, mojibake, text containing
NUL and some long URLs can now be flagged, and ``.strategies`` and ``.weights``
have six entries. ``profile="legacy"`` selects the three 0.8.0 members:

.. code-block:: python

   from pygarble import EnsembleDetector

   current = EnsembleDetector()
   former_members = EnsembleDetector(profile="legacy")
   assert current.predict("qwerty") is True
   assert [s.name for s in former_members.strategies] == [
       "MARKOV_CHAIN", "LOG_LIKELIHOOD_RATIO", "WORD_ANOMALY"
   ]

``legacy`` restores the 0.8.0 membership, not the 0.8.0 output: the shared
tokeniser and the correctness fixes apply to it too.

Weighted voting with three weights on the default set now raises
``ValueError``, because the default set has six members. The message gives both
lengths. Select the legacy profile to keep three weights:

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector(
       profile="legacy", voting="weighted", weights=[2.0, 1.0, 1.0]
   )
   assert len(detector.weights) == 3

Long tokens are not garbled by length alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 0.8.0 every strategy scored text with a single token over 1,000 characters
as 1.0 and garbled. That rule is now opt-in: pass ``max_string_length=1000`` to
classify long non-URL tokens as garbled again. To reject oversized input with
``ValueError`` instead, set ``max_input_length``.

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   long_token = "abc" * 700
   default = GarbleDetector(Strategy.SYMBOL_RATIO, threshold=1.0)
   old_rule = GarbleDetector(
       Strategy.SYMBOL_RATIO, threshold=1.0, max_string_length=1000
   )
   assert default.predict(long_token) is False
   assert old_rule.predict(long_token) is True

Empty input and ``threshold=0.0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Empty or wholly inapplicable input predicts ``False``, and ``analyze()``
reports ``insufficient_evidence``. ``applicable("")`` and ``applicable("!!!")``
return ``False``. ``threshold=0.0`` therefore no longer flags every non-empty
text.

Exceptions and newly rejected values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check any ``except`` clause around detector construction:

.. list-table::
   :header-rows: 1

   * - Input
     - 0.8.0
     - 0.11.0
   * - Wrongly typed ``threshold``, ``threads``, ``threshold_per_char`` or weight
     - ``TypeError``
     - ``ValueError``
   * - Huge integer weight such as ``10**400``
     - ``OverflowError``
     - ``ValueError``
   * - Unknown strategy name
     - ``AttributeError``
     - ``ValueError``
   * - ``None`` or a number as the strategy
     - ``AttributeError``
     - ``TypeError``
   * - ``EnsembleDetector(strategies=[None])``
     - ``AttributeError``
     - ``ValueError`` naming the bad member
   * - Invalid custom regex for ``pattern_matching``
     - ``re.error``
     - ``ValueError`` chained from ``re.error``
   * - A bare string where a list of names belongs
     - varies
     - an error that is both ``TypeError`` and ``ValueError``

Some values that 0.8.0 accepted now raise. ``threshold=True``,
``threads=2.5``, NaN weights and a non-bool ``override_defaults`` raise
``ValueError``. So does a huge integer such as ``10**400`` for any numeric
option: in 0.8.0 ``threads``, ``timeout_per_text``, ``max_string_length`` and
``min_length`` accepted it, as did settings 0.8.0 ignored, such as ``n``. ``applicable(None)`` and a batch with a non-string entry, such
as ``predict([None])``, raise ``TypeError``; batches are validated before any
work starts. Worker errors and timeouts propagate. ``timeout_per_text`` bounds
the wait for threaded results; it is not a hard execution deadline.

Future warnings
~~~~~~~~~~~~~~~

Two settings that 0.8.0 accepted silently now emit ``FutureWarning`` at your
call site and will become errors in a future release:

* Unknown strategy settings. Check the accepted names in :doc:`strategies`.
* ``weights`` with a voting mode other than ``weighted``, where they are
  ignored.

Test suites run with ``-W error`` or ``filterwarnings = error`` fail on these.
Remove the setting, or filter ``FutureWarning`` from ``pygarble`` while you
migrate. ``EnsembleDetector`` forwards shared keyword arguments only to members
that accept them, with one warning for a key that no selected member accepts.
Use ``strategy_kwargs`` for per-member settings, including a strategy's own
``threshold``, which is independent of the detector's decision threshold.

More strategies
~~~~~~~~~~~~~~~

``Strategy`` and ``STRATEGY_MAP`` have 32 entries instead of 26. Code that
iterates either to build "all strategies" also gets ``CONTROL_CHARACTERS``,
``LOCAL_ANOMALY``, ``CROSS_PARSING``, ``PRIMED_COMPRESSION``, ``NGRAM_RANK``
and ``PERMUTATION_TEST``. ``STRATEGY_MAP`` is still a plain ``dict``, and
detectors use its entries as in 0.8.0.

Deprecated import paths
~~~~~~~~~~~~~~~~~~~~~~~

These 0.8.0 imports still work but emit a ``DeprecationWarning`` naming the
canonical home. They will be removed in a future release.

.. list-table::
   :header-rows: 1

   * - 0.8.0 import
     - Canonical import
   * - ``from pygarble.core import BaseStrategy`` and the 26 strategy classes
     - ``from pygarble.gibberish.strategies import ...``
   * - ``BIGRAM_LOG_PROBS`` and ``DEFAULT_LOG_PROB`` from
       ``pygarble.strategies.markov_chain``, ``word_anomaly`` or
       ``log_likelihood_ratio``
     - ``from pygarble.data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB``
   * - ``from pygarble.strategies.zipf_conformity import FunctionWordDensityStrategy``
     - ``from pygarble.gibberish.strategies import FunctionWordDensityStrategy``

``pygarble.core`` no longer re-exports incidental names that 0.8.0 exposed
through its own imports, such as the ``typing`` helpers ``Callable`` and
``Dict`` and the ``concurrent`` module. Import them from the standard library.
``from pygarble import GarbleDetector, EnsembleDetector, Strategy`` works
without warnings.

Pickles
~~~~~~~

``GarbleDetector`` and ``EnsembleDetector`` objects pickled by 0.8.0 load and
predict in 0.11.0; a 0.8.0 default ensemble loads as the ``legacy`` profile.
They then score with 0.11.0 behaviour. Pickles created by 0.11.0 use the new
module paths and are not guaranteed to load in 0.8.0.

Scores and settings
~~~~~~~~~~~~~~~~~~~

* ``score`` is a new alias of ``predict_proba``; neither is a calibrated
  probability.
* ``WORD_LOOKUP``'s ``unknown_threshold`` now affects the score. Its default 0.5
  keeps the 0.8.0 mapping; other values change results.
* Calling a strategy's ``predict`` or ``predict_proba`` directly scores text the
  strategy deems not applicable as 0.0.
* Invalid numeric settings, including nonfinite weights and huge integers,
  raise ``ValueError``.
* Majority voting uses a strict majority of applicable decisions; its reported
  score is their mean and need not cross the threshold with the same result.
* Meaningful Hindi and other non-English text scoring as gibberish remains
  expected.

Module layout
~~~~~~~~~~~~~

0.11.0 adds two focused modules: ``pygarble.gibberish`` for gibberish detection
and ``pygarble.screening`` for secrets, PII and profanity. Top-level imports and
old module paths point to the same implementations, so classes, enums and
caches are shared and mixing old and new imports is supported. See
:doc:`architecture` for the complete layout.

.. code-block:: python

   from pygarble import EnsembleDetector as ExistingDetector
   from pygarble.gibberish import EnsembleDetector
   from pygarble.pii import PIIDetector as ExistingPII
   from pygarble.screening import PIIDetector

   assert ExistingDetector is EnsembleDetector
   assert ExistingPII is PIIDetector

Two scanners are available. Choose one deliberately:

.. code-block:: python

   from pygarble import Scanner as CombinedScanner
   from pygarble.screening import Scanner

   text = "qxzjkwpv bnmqwer zzxqv"
   assert CombinedScanner().scan(text).flagged
   assert not Scanner().scan(text).flagged

``pygarble.Scanner`` runs all four categories by default.
``pygarble.screening.Scanner`` runs secrets, PII and profanity, and accepts
optional backends and custom detectors. It does not accept gibberish
``profile``, ``threshold`` or ``allowlist`` settings; use the gibberish API
separately, or use the combined scanner. The standalone
``profanity_allowlist`` setting configures profanity exceptions.

Finding and redaction types are shared. Both scanners support ``scan``,
``scan_batch``, ``iter_scan`` and ``redact``. The standalone convenience
functions construct a scanner per call, while the top-level functions keep a
configuration cache. Reuse a scanner for repeated work, particularly when
optional backends are enabled.

Command line and optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0.8.0 had no command line. 0.11.0 adds two:

* ``pygarble`` (also ``python -m pygarble``) reads one text per line:
  ``check``, ``score``, ``analyze``, ``calibrate``, ``scan`` and ``redact``.
  ``scan`` output includes the input text. See :doc:`cli`.
* ``pygarble-screen`` (also ``python -m pygarble.screening``) reads whole
  documents, omits source text from scan JSON unless ``--include-text`` is set,
  and redacts multiline private keys. See :doc:`standalone-screening`.

The base install remains dependency-free. Extras provide phonenumberslite,
python-stdnum and detect-secrets; Gitleaks is installed separately. Installing
these tools alone does not change results: select ``backends`` explicitly.
Their findings may extend native coverage or change which overlapping kind
labels a redaction. The JSON data tables under ``pygarble/data/`` ship in the
repository and source distribution only, not the wheel, and the detector does
not read them at runtime.

What is new
~~~~~~~~~~~

* Screening for secrets, PII and profanity with redaction:
  :doc:`screening`, :doc:`secrets`, :doc:`pii`, :doc:`profanity` and
  :doc:`standalone-screening`.
* Profiles (``english``, ``english_extended``, ``legacy``, ``corruption``,
  ``spoofing``, ``llm_output``, ``english_fusion``), ``analyze()`` with spans,
  allowlists, ``strategy_kwargs`` and ``max_input_length``: :doc:`api` and
  :doc:`strategy-guide`.
* Fisher voting and four English-reference strategies: :doc:`strategy-guide`.
* Threshold calibration from labeled samples: :doc:`calibration`.
* The command lines: :doc:`cli`.
