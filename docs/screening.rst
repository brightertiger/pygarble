Screening and redaction
=======================

:class:`pygarble.Scanner` runs every enabled category over a text and
returns a :class:`pygarble.ScanReport`. Findings say what was found and
where; they never contain the matched text, so a logged report cannot leak
a secret.

.. code-block:: python

   from pygarble import Scanner

   scanner = Scanner()
   report = scanner.scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE")
   assert report.flagged
   assert report.kinds() == ("aws_access_key_id", "email")
   assert scanner.redact("mail jane@example.com").text == "mail [EMAIL]"

The module-level :func:`pygarble.scan` and :func:`pygarble.redact` accept the
same keyword arguments and reuse a cached scanner per configuration.
``scan_batch`` screens a list and ``iter_scan`` screens any iterable lazily.

Each :class:`pygarble.Finding` has a ``category``, ``kind``, ``start`` and
``end`` (Python string offsets, end exclusive), a ``confidence`` and a short
``reason``. :class:`pygarble.ScanReport` groups findings with
``by_category()``, lists distinct kinds with ``kinds()`` and serialises with
``to_dict()``.

.. code-block:: python

   from pygarble import scan

   report = scan("call +14155550123 now")
   assert report.to_dict() == {
       "flagged": True,
       "length": 21,
       "findings": [
           {
               "category": "pii",
               "kind": "phone",
               "start": 5,
               "end": 17,
               "confidence": 0.9,
               "reason": "e164",
           }
       ],
   }

Categories
----------

``secrets``, ``pii``, ``profanity`` and ``gibberish``. Pass a subset to the
constructor to run fewer. ``kinds`` and ``exclude_kinds`` select individual
kinds across categories; ``locales`` selects PII locale packs (``us``,
``uk``, ``in``); ``profile``, ``threshold`` and ``allowlist`` configure the
gibberish category exactly like :class:`pygarble.EnsembleDetector`.
``profanity_allowlist`` exempts words from the profanity check,
``secrets_without_context=True`` enables standalone high-entropy strings, and
``max_input_length`` rejects oversized input with ``ValueError``.

.. code-block:: python

   from pygarble import Scanner

   pii_only = Scanner(categories=["pii"], locales=["uk"])
   assert pii_only.scan("key AKIAIOSFODNN7EXAMPLE").findings == ()
   assert pii_only.scan("NI AB123456C").kinds() == ("nino",)

The gibberish category reports at most one finding, kind ``garbled``,
spanning the whole text, and only when the ensemble decides the text is
garbled. Its confidence is the ensemble score. It is never redacted.

Confidence tiers
----------------

Each rule has a fixed confidence. ``flagged`` is true when any finding
reaches ``min_confidence`` (default 0.5) or when a gibberish finding exists;
gibberish is gated by ``threshold`` instead. Findings below
``min_confidence`` are still returned.

.. list-table::
   :header-rows: 1

   * - Tier
     - Meaning
   * - 1.0
     - checksum verified or unique vendor prefix
   * - 0.9
     - unambiguous structure without a checksum
   * - 0.8
     - structural with some ambiguity; elongated, embedded or spaced
       strong profanity
   * - 0.7
     - weak or context dependent (IP addresses, mild profanity)
   * - 0.6
     - keyword plus entropy, or ambiguous masking
   * - 0.5
     - standalone high-entropy strings (opt-in)
   * - score
     - the gibberish ensemble score

Redaction
---------

``redact`` replaces every finding at or above ``min_confidence`` in the
chosen categories (all but gibberish by default) and returns a
:class:`pygarble.Redaction` with the new ``text``, the ``findings`` it
replaced and a ``count``. Overlapping findings become one region, labelled
with the highest-confidence kind. Modes:

* ``placeholder`` (default): ``[EMAIL]``. The template may use only the bare
  fields ``{KIND}``, ``{kind}`` and ``{category}``; it is validated before
  any output.
* ``mask``: every character replaced by ``mask_char``, preserving length.
* ``partial``: like ``mask``, but keeps the last four characters when the
  region ends with a ``credit_card``, ``phone``, ``iban``, ``ssn_us``,
  ``nhs_number`` or ``aadhaar`` finding and no other kind covers them.

.. code-block:: python

   from pygarble import redact

   assert redact("card 4111 1111 1111 1111", mode="partial").text == (
       "card ***************1111"
   )
   assert redact("mail jane@example.com", mode="mask").text == (
       "mail ****************"
   )
   assert redact("mail jane@example.com", placeholder="<{category}>").text == (
       "mail <pii>"
   )

The ``pygarble scan`` and ``pygarble redact`` commands expose the same
options; see :doc:`cli`. Category details are in :doc:`secrets`, :doc:`pii`
and :doc:`profanity`.

What it does not catch
----------------------

Names, postal addresses, free-text dates of birth, hate speech beyond a
word list, non-English profanity, and secrets without a recognisable shape.
Send those to a model after this pass.
