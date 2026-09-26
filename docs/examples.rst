Python examples
===============

These examples use the current source APIs, including unreleased module
paths; see :doc:`installation`. Each example
includes its own imports and data. Reuse a configured detector across calls.

Redact a batch without gibberish checks
---------------------------------------

.. code-block:: python

   from pygarble.screening import Scanner

   scanner = Scanner(categories=["secrets", "pii"], max_input_length=10_000)
   texts = ["mail jane@example.com", "key AKIAIOSFODNN7EXAMPLE", "hello"]
   reports = scanner.scan_batch(texts)
   assert [report.flagged for report in reports] == [True, True, False]
   redacted = [scanner.redact(text).text for text in texts]
   assert redacted == ["mail [EMAIL]", "key [AWS_ACCESS_KEY_ID]", "hello"]
   assert scanner.scan("qxzjkwpv bnmqwer zzxqv").flagged is False

For a large input stream, ``scanner.iter_scan(iterable)`` yields reports
one at a time. It does not split a document or materialize the input iterable.

Configure native screening rules
--------------------------------

.. code-block:: python

   from pygarble.screening import Scanner

   scanner = Scanner(
       categories=["pii", "profanity"],
       locales=["uk"],
       profanity_allowlist=["damn"],
   )
   assert scanner.scan("NI AB123456C").kinds() == ("nino",)
   assert scanner.scan("damn").findings == ()

``locales`` selects native PII packs. Optional phone-number backends have
their own ``region`` option. See :doc:`standalone-screening` for complete
backend examples and :doc:`pii`, :doc:`profanity` and :doc:`secrets` for rules.

Validate a required English field
---------------------------------

.. code-block:: python

   from typing import Optional
   from pygarble.gibberish import EnsembleDetector

   detector = EnsembleDetector(max_input_length=10_000)

   def validate_english_field(text: str) -> Optional[str]:
       if not text.strip():
           return "Enter some English text."
       if len(text) > 10_000:
           return "Use at most 10,000 characters."
       if detector.predict(text):
           return "Please review this text; it does not pass the English checks."
       return None

   assert validate_english_field("Hello world") is None
   assert validate_english_field("") == "Enter some English text."
   assert validate_english_field("asdfghjkl") is not None

Use this policy for fields that expect English. Non-English text may be meaningful
and still fail these checks. Rare words and names may also require review.

Partition a dataset for review
------------------------------

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector

   texts = ["Hello world", "asdfghjkl", "Please review this text"]
   detector = EnsembleDetector()
   decisions = detector.predict(texts)
   flagged = [text for text, bad in zip(texts, decisions) if bad]
   remaining = [text for text, bad in zip(texts, decisions) if not bad]
   assert flagged == ["asdfghjkl"]
   assert len(remaining) == 2

For larger sources, submit lists in caller-controlled chunks. The returned lists
are materialized, and input validation covers each submitted batch.

Export explanations as JSON
---------------------------

.. code-block:: python

   import json
   from dataclasses import asdict
   from pygarble.gibberish import GarbleDetector, Strategy

   text = "Please review qxzjkwpvm before delivery."
   result = GarbleDetector(Strategy.LOCAL_ANOMALY).analyze(text)
   assert result.garbled is True
   assert any(text[s.start:s.end] == "qxzjkwpvm" for s in result.spans)
   payload = json.dumps(asdict(result), ensure_ascii=False)
   assert json.loads(payload)["status"] == "garbled"

``asdict`` includes spans under each signal. ``result.spans`` is a convenience
property combining those spans, not an additional serialized dataclass field.
Offsets use Python string indices, not byte offsets; the end is exclusive.

Check encoding independently of language
----------------------------------------

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector

   detector = EnsembleDetector(profile="corruption")
   texts = ["नमस्ते दुनिया", "Café au lait", "CafÃ© au lait", "hello\x00world"]
   assert detector.predict(texts) == [False, False, True, True]

This detects selected artifacts; it does not repair encoding or certify that
all possible encoding problems have been found.

Configure members independently
-------------------------------

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector, Strategy

   detector = EnsembleDetector(
       strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
       voting="weighted",
       weights=[0.7, 0.3],
       threshold=0.5,
       strategy_kwargs={
           Strategy.MARKOV_CHAIN: {"min_length": 4},
           Strategy.WORD_ANOMALY: {"min_word_length": 6},
       },
       allowlist=["syzygy"],
   )
   assert detector.predict("syzygy") is False

A custom strategy list defaults to majority voting unless overridden. Validate
thresholds and weights against representative inputs; they are policy choices,
not calibrated confidence levels.
