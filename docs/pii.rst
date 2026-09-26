PII
===

:class:`pygarble.screening.PIIDetector` finds personal data by structure and, where the
format has one, a checksum. Findings carry the kind, span, confidence and
reason, never the value.

.. code-block:: python

   from pygarble.screening import PIIDetector

   detector = PIIDetector()
   report = detector.detect("call +14155550123 or mail jane@example.com")
   assert [(f.kind, f.confidence) for f in report] == [
       ("phone", 0.9),
       ("email", 0.9),
   ]

   uk = PIIDetector(locales=["uk"])
   assert [f.kind for f in uk.detect("NI AB123456C, NHS 943 476 5919")] == [
       "nino",
       "nhs_number",
   ]
   assert detector.detect("icon name@2x.png") == ()

Kinds
-----

.. list-table::
   :header-rows: 1
   :widths: 18 10 52 20

   * - Kind
     - Locale
     - Shape
     - Confidence
   * - ``email``
     - all
     - ``local@domain.tld``; file-extension TLDs such as ``name@2x.png``
       are excluded
     - 0.9
   * - ``phone``
     - all
     - E.164 (``+`` and country code)
     - 0.9
   * - ``credit_card``
     - all
     - 13 to 19 digits with a known brand prefix and a valid Luhn check
     - 1.0
   * - ``iban``
     - all
     - country code, valid mod-97 check and the country's length
     - 1.0
   * - ``ipv4``, ``ipv6``
     - all
     - address structure
     - 0.7
   * - ``phone``
     - us, uk, in
     - national formats; US numbers follow the NANP rule
     - 0.8
   * - ``ssn_us``
     - us
     - ``123-45-6789`` with separators (bare nine digits need a keyword: 0.6)
     - 0.8
   * - ``nino``
     - uk
     - National Insurance number, ``AB123456C``
     - 0.9
   * - ``nhs_number``
     - uk
     - ten digits with a valid mod-11 check; 0.9 with a keyword nearby, 0.8
       when spaced without one
     - 0.8 / 0.9
   * - ``aadhaar``
     - in
     - twelve digits with a valid Verhoeff check
     - 1.0
   * - ``pan``
     - in
     - Permanent Account Number, ``ABCPE1234F``
     - 0.9

Locale packs
------------

``locales`` defaults to ``("us", "uk", "in")``. Each pack adds its national
phone format and identifiers: ``us`` adds ``ssn_us``; ``uk`` adds ``nino``
and ``nhs_number``; ``in`` adds ``aadhaar`` and ``pan``. Pass ``kinds`` or
``exclude_kinds`` to select individual rules.

Matching rules
--------------

* Bare NHS and Aadhaar digit runs need a keyword nearby.
* Digit rules stop at word characters, so a number glued to letters, such
  as ``order12345678``, is not matched.
* A phone outranks a keyword-less NHS number on the same span, and a
  keyword-less NHS number nested inside a phone is dropped.

Names, postal addresses and free-text dates of birth are not detected.

Implementation and optional coverage
------------------------------------

Native patterns and checksums live in ``pygarble/screening/pii/``. Existing
``pygarble.pii`` imports remain compatibility pointers to the same objects.
The ``phonenumbers`` backend adds numbering-plan validation; ``stdnum`` adds
selected identifier validators. Enable them on
:class:`pygarble.screening.Scanner`; constructing ``PIIDetector`` continues
to run native rules only. See :doc:`standalone-screening` for supported
formats and the difference between native ``locales`` and backend ``region``.
