Profanity
=========

:class:`pygarble.ProfanityDetector` finds English profanity at the token
level, so words that merely contain a profane substring stay clean. Every
finding has kind ``profanity``; the confidence tells strong words from mild
ones and plain spellings from obfuscated ones.

.. code-block:: python

   from pygarble import ProfanityDetector

   detector = ProfanityDetector()
   assert detector.detect("Scunthorpe assassin classic bass") == ()

   (mild,) = detector.detect("what a damn mess")
   assert (mild.kind, mild.confidence, mild.reason) == (
       "profanity",
       0.7,
       "mild",
   )
   assert [f.reason for f in detector.detect("f u c k this")] == ["spaced"]
   assert ProfanityDetector(allowlist=["damn"]).detect("damn") == ()

Tiers and obfuscation
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Form
     - Example
     - Confidence
   * - strong word, including leetspeak
     - ``sh1t``
     - 1.0
   * - unambiguous masking
     - ``f*cking``
     - 0.9
   * - elongated, embedded or spaced strong word
     - ``fuuuck``, ``f u c k``
     - 0.8
   * - mild word, including elongated or spaced forms
     - ``damn``, ``crap``
     - 0.7
   * - ambiguous masking (a dictionary word also fits)
     - ``f*ck``, ``sh*t``
     - 0.6

The mild tier also holds words with innocent senses, such as ``cum``,
``prick``, ``cock``, ``tits`` and ``pissed``. Embedded matching applies only
to compounds of "fuck". Multi-word phrases are matched as phrases.
Digit-only tokens are ignored; possessives and trailing ``!`` are handled.
``allowlist`` exempts words, and ``tiers`` selects ``strong``, ``mild`` or
both.

The detector covers English only and does not detect hate speech beyond its
word list.

Word list attribution
---------------------

The word list is seeded from the LDNOOBW English list (List of Dirty,
Naughty, Obscene, and Otherwise Bad Words) published by Shutterstock under
CC-BY-4.0, then filtered and extended by the pygarble maintainers. The
attribution text ships as ``pygarble.profanity.wordlist.ATTRIBUTION``.
