Standalone screening and optional backends
==========================================

``pygarble.screening`` screens secrets, PII and profanity. It does not run
the gibberish ensemble. It shares the existing finding and redaction types,
so applications can use either API without translating results.

.. code-block:: python

   from pygarble.screening import Scanner

   scanner = Scanner(max_input_length=100_000)
   assert not scanner.scan("qxzjkwpv bnmqwer zzxqv").flagged
   assert scanner.redact("mail jane@example.com").text == "mail [EMAIL]"

The existing ``from pygarble import Scanner`` and ``pygarble scan`` retain
their four-category defaults for compatibility. Existing imports of
``PIIDetector``, ``SecretsDetector`` and ``ProfanityDetector`` remain valid;
they are also exported from ``pygarble.screening``. Detector implementations
and collection/redaction logic are shared between the two APIs.

See :doc:`architecture` for source layout and compatibility pointers, and
:doc:`api` for the scanner, detector and backend signatures. These APIs are
part of the unreleased source changes described in :doc:`migration`.

Installation and explicit selection
------------------------------------

The base install remains dependency-free. Optional backends supplement
native rules only when selected; installing an extra alone changes nothing.
For these unreleased features, install from the source checkout described in
:doc:`installation`:

.. code-block:: console

   python -m pip install -e '.[phones]'     # phonenumberslite
   python -m pip install -e '.[stdnum]'     # python-stdnum
   python -m pip install -e '.[secrets]'    # detect-secrets
   python -m pip install -e '.[screening]'  # all three Python dependencies

Install Gitleaks >=8.19,<9 separately from its official distribution and
put it on PATH, or supply its executable path. CI exercises Gitleaks 8.30.1.
The adapter does not download or install executables.

The following example requires the optional packages and Gitleaks:

.. code-block:: python

   from pygarble.screening import Scanner

   document = "Call +442079460958 or mail jane@example.com."
   scanner = Scanner(
       backends=["phonenumbers", "stdnum", "detect-secrets", "gitleaks"],
       backend_options={
           "phonenumbers": {"region": "GB"},
           "stdnum": {"formats": ["iban", "br.cpf", "de.idnr"]},
           "gitleaks": {"timeout": 5.0},
       },
       max_input_length=100_000,
   )
   report = scanner.scan(document)
   safe_text = scanner.redact(document).text
   assert report.flagged
   assert "jane@example.com" not in safe_text

Reuse an instance for repeated calls. ``scan_batch`` accepts a sequence and
``iter_scan`` processes an iterable lazily. The module-level ``scan`` and
``redact`` convenience functions construct a scanner per call.

Backends and their limits
-------------------------

``phonenumbers``
   Uses ``phonenumberslite`` to extract and validate international phone
   formats with original character offsets. ``region`` is an optional ISO
   country code, such as ``GB`` or ``IN``, for national numbers; international
   numbers can be found without it. This region is separate from native
   ``locales``. It loads numbering metadata, not an NLP model. The full
   ``phonenumbers`` package also works, but do not install both distributions
   into the same environment because they provide the same import namespace.

``stdnum``
   Runs curated candidate patterns and local ``is_valid`` functions from
   ``python-stdnum``. Supported ``formats`` are ``iban``, ``in_.pan``,
   ``in_.aadhaar``, ``us.ssn``, ``gb.nhs``, ``br.cpf`` and ``de.idnr``;
   all are selected by default, independently of native ``locales``.
   New kinds are ``cpf_br`` and ``idnr_de``. Bare Aadhaar, NHS, CPF and
   German tax identifiers require context labels in this adapter; punctuated
   CPF is also recognized. The library's other validators are not automatically
   free-text detectors. Its license is LGPL-2.1-or-later; it is an optional
   dependency, and its implementation is not copied into pygarble.

``detect-secrets``
   Runs local string plugins, with no verification calls, global settings
   changes or model-based filters. Findings have kind ``detect_secrets_secret``
   and a reason such as ``detect_secrets:GitHubTokenDetector``. The default
   excludes standalone entropy, public IPs and the header-only private-key
   plugin. Native rules handle complete private keys. ``plugins`` can select
   upstream class names, for example ``["KeywordDetector", "AWSKeyDetector"]``.
   Quoted weak passwords can be detected by ``KeywordDetector``; arbitrary
   unquoted passwords are not guaranteed. Keyword values repeated on the same
   line are conservatively redacted together. Entropy plugins can be selected
   explicitly and retain their upstream entropy thresholds.

``gitleaks``
   Scans one document through stdin per subprocess. Findings have kind
   ``gitleaks_secret`` and reasons such as ``gitleaks:github-pat``. UTF-8 byte
   coordinates are translated to Python character offsets and checked against
   the original value before redaction. The adapter isolates ambient config,
   ignores inline allow comments and disables archive/encoding expansion.
   Options are ``executable``, a positive ``timeout`` (seconds, default 10),
   and an optional explicit ``config`` file. Reports temporarily contain
   source values in a private directory, removed after the call; subprocess
   output is never forwarded. Batch documents rather than launching Gitleaks
   separately for each short line.

All backends are additive. A validator does not veto a native finding.
Use ``builtin=False`` for only optional/custom detectors, or compose the
detector classes exported from ``pygarble.screening.backends`` explicitly.
Turning off native rules also turns off their coverage, including complete
private-key handling when only detect-secrets is selected.

``kinds`` and ``exclude_kinds`` apply across all enabled detectors. Identical
category/kind/spans keep the strongest evidence; overlapping different kinds
remain visible, and redaction merges them. Confidence values are evidence
tiers, not probabilities or confirmation that a credential is active.

Profanity remains the native English word-list engine. Configure
``profanity_tiers=["strong"]`` and ``profanity_allowlist`` as needed. This
change adds no contextual toxicity classifier or multilingual profanity model.

Runtime cost
-------------

A local smoke measurement on Python 3.12/macOS ARM64 used a synthetic,
approximately 1 KB document containing repeated English prose and an email.
After warm-up, median scan times were:

.. list-table::
   :header-rows: 1

   * - Configuration
     - Milliseconds per document
   * - Native rules
     - 0.16
   * - Native + phonenumbers
     - 0.21
   * - Native + stdnum
     - 0.40
   * - Native + detect-secrets
     - 1.39
   * - Native + Gitleaks
     - 283

These are 100 repeated scans per configuration (30 for Gitleaks), excluding
Python scanner construction but including each Gitleaks subprocess. They are
not production latency guarantees or detection-quality measurements. Package
versions are recorded in the development plan. Keep native rules as the cheap
default; enable optional providers where their additional coverage is needed.
Gitleaks is better suited to larger documents or a separate bulk pass than
to a process launch for each chat message.

Errors and resource limits
--------------------------

Missing dependencies fail during construction with installation guidance.
Backend execution errors, timeouts and invalid spans raise ``BackendError``;
they never return a clean result. Error messages do not include scanned text.
An application must treat an error as an incomplete scan.

Set ``max_input_length`` to bound document size; the Python API does not
choose a limit for the application. Only the Gitleaks subprocess has a hard
timeout. Python detectors run synchronously. No runtime network verification
or model downloads are performed. Real-package tests block Python socket and
DNS calls and separately exercise the installed Gitleaks executable.

Names, addresses, arbitrary passwords and contextual abuse still need other
handling. "No findings" means no selected rule matched, not a safety guarantee.

Document CLI
-------------

``pygarble-screen`` and ``python -m pygarble.screening`` accept whole UTF-8
documents from files or stdin. Whole-document redaction preserves multiline
private-key detection. Scan output is one JSON report per document with
findings and input length; source text is included only with ``--include-text``.

.. code-block:: console

   pygarble-screen scan document.txt
   pygarble-screen redact document.txt
   pygarble-screen scan --backends phonenumbers,stdnum document.txt
   pygarble-screen scan --backends phonenumbers \
       --backend-options '{"phonenumbers":{"region":"GB"}}' document.txt

The CLI defaults to at most 1,000,000 characters per document, configurable
with ``--max-input-length``. Exit codes: 0 for a clean scan or successful
redaction, 1 for a flagged scan, 2 for configuration/input/backend errors.
The legacy ``pygarble scan`` remains line-oriented and echoes source text;
use the new CLI for document redaction and findings-only logs.

Custom detectors
-----------------

Supply per-instance detectors implementing ``ScreeningDetector``. Their
``category`` must be secrets, PII or profanity, ``kinds`` declares supported
names, and ``detect`` returns findings in original Python character offsets.
Reasons must be static descriptions, never copies of matched values.

.. code-block:: python

   from pygarble.screening import Finding, Scanner

   class CustomerDetector:
       category = "pii"
       kinds = frozenset({"customer_id"})

       def detect(self, text):
           start = text.find("ID123")
           if start < 0:
               return ()
           return (Finding("pii", "customer_id", start, start + 5,
                           0.9, "customer_identifier"),)

   scanner = Scanner(detectors=[CustomerDetector()])
   assert scanner.redact("customer ID123").text == "customer [CUSTOMER_ID]"

Backend sources
----------------

* `python-phonenumbers <https://github.com/daviddrysdale/python-phonenumbers>`_
* `python-stdnum <https://arthurdejong.org/python-stdnum/>`_
* `detect-secrets <https://github.com/Yelp/detect-secrets>`_
* `Gitleaks <https://github.com/gitleaks/gitleaks>`_
