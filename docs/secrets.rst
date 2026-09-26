Secrets
=======

:class:`pygarble.screening.SecretsDetector` finds credentials by shape. Findings carry
the kind, span, confidence and reason, never the secret itself.

.. code-block:: python

   from pygarble.screening import SecretsDetector

   detector = SecretsDetector()
   (finding,) = detector.detect("token ghp_\u00611B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8")
   assert finding.kind == "github_token"
   assert finding.confidence == 1.0

   (generic,) = detector.detect("DB_PASSWORD=Xk9#mQ2vLp8zRt")
   assert (generic.kind, generic.confidence) == ("generic_secret", 0.6)
   assert detector.detect("password = changeme") == ()

Kinds
-----

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Kind
     - Shape
     - Confidence
   * - ``aws_access_key_id``
     - ``AKIA``, ``ASIA``, ``ABIA`` or ``ACCA`` plus 16 characters
     - 1.0
   * - ``aws_secret_access_key``
     - 40 base64 characters after an ``aws ... secret`` or ``key`` keyword
     - 0.9
   * - ``github_token``
     - ``ghp_``, ``gho_``, ``ghu_``, ``ghs_``, ``ghr_``, ``github_pat_``
     - 1.0
   * - ``gitlab_token``
     - ``glpat-``
     - 1.0
   * - ``slack_token``
     - ``xoxa-``, ``xoxb-``, ``xoxp-``, ``xoxr-``, ``xoxs-``
     - 1.0
   * - ``slack_webhook``
     - ``https://hooks.slack.com/services/...``
     - 1.0
   * - ``stripe_key``
     - ``sk_live_`` or ``rk_live_`` (``_test_`` keys 0.8)
     - 1.0
   * - ``google_api_key``
     - ``AIza`` plus 35 characters
     - 1.0
   * - ``openai_api_key``
     - ``sk-...T3BlbkFJ...`` (legacy ``sk-`` plus 48 characters 0.9)
     - 1.0
   * - ``anthropic_api_key``
     - ``sk-ant-api`` or ``sk-ant-admin``
     - 1.0
   * - ``huggingface_token``
     - ``hf_`` plus 34 characters
     - 1.0
   * - ``npm_token``
     - ``npm_`` plus 36 characters
     - 1.0
   * - ``pypi_token``
     - ``pypi-AgEIcHlwaS5vcmc...``
     - 1.0
   * - ``sendgrid_key``
     - ``SG.`` key
     - 1.0
   * - ``jwt``
     - three base64url segments, the first two starting ``eyJ``
     - 1.0 (0.8 when the header does not decode)
   * - ``private_key``
     - a PEM or PGP ``BEGIN ... PRIVATE KEY`` block (not certificates)
     - 1.0
   * - ``url_credentials``
     - ``scheme://user:password@host``
     - 0.9
   * - ``bearer_token``
     - ``Bearer`` followed by 20 or more token characters
     - 0.8
   * - ``generic_secret``
     - a keyword (``password``, ``token``, ``api_key``, ``DB_PASSWORD``...)
       with a high-entropy value
     - 0.6
   * - ``high_entropy_string``
     - a standalone high-entropy string; only with ``without_context=True``
     - 0.5

Generic secrets
---------------

A generic secret needs a keyword, including identifiers such as
``DB_PASSWORD`` or ``api_key_prod``, and an assigned value with enough
entropy. Values of 8 to 22 characters need at least 3.0 bits per character
and a mix of character classes. Longer values need 3.0 bits per character
for hex, 4.5 for base64 and 3.5 otherwise. Placeholders such as
``changeme`` or ``<your-token>`` are ignored, and a generic secret that
overlaps a known-prefix finding is dropped.

A vendor token or JWT inside a ``bearer_token`` or ``url_credentials``
value is also reported on its own, as a second finding nested inside the
first, so ``Authorization: Bearer ghp_...`` yields both ``bearer_token``
and ``github_token``. Redaction merges the two into one region.

Pass ``kinds`` or ``exclude_kinds`` to select rules. Selection only filters:
the findings of a kind are the same whether or not other kinds are
selected, and an unselected rule never hides a selected one. In a
:class:`pygarble.screening.Scanner`, ``secrets_without_context=True`` enables
``high_entropy_string``.

The test vectors in ``pygarble/data/secrets.json`` are split into
8-character chunks (``"vectors_encoding": "chunks8"``) so secret scanners
do not flag them; a port concatenates each list.

Implementation and optional scanners
------------------------------------

Native patterns and entropy checks live in ``pygarble/screening/secrets/``;
old ``pygarble.secrets`` imports remain compatibility pointers. Use
:class:`pygarble.screening.Scanner` to add ``detect-secrets`` or ``gitleaks``
explicitly. These backends emit their own kinds and supplement native rules;
they do not verify whether a credential is active. Native rules still provide
complete multiline private-key spans. See :doc:`standalone-screening` for
backend options, errors and subprocess cost.
