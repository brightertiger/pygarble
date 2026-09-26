Command-line interface
======================

The package provides two command-line entry points:

.. list-table::
   :header-rows: 1

   * - Command
     - Input and purpose
   * - ``pygarble-screen`` / ``python -m pygarble.screening``
     - Whole UTF-8 documents; secrets, PII and profanity
   * - ``pygarble`` / ``python -m pygarble``
     - One text per line; gibberish commands and the compatible combined scanner

The document CLI is part of the unreleased source changes; see
:doc:`installation`. The existing line-oriented commands keep their behavior.

Screen or redact a document
---------------------------

.. code-block:: bash

   python -m pygarble.screening scan document.txt
   python -m pygarble.screening redact document.txt
   printf 'mail jane@example.com\n' | pygarble-screen redact --mode mask

``scan`` emits one JSON report per file or stdin document, with findings and
offsets but no source text. ``--include-text`` explicitly adds the original
document. ``redact`` writes transformed text, preserving line endings and
handling private-key blocks across lines. When processing multiple files,
redacted outputs are concatenated without adding separators.

Inputs default to stdin; ``-`` also names stdin. The CLI reads at most
``--max-input-length`` Python characters (default 1,000,000), and rejects
oversized documents and invalid UTF-8. The Python scanner's input limit is
``None`` by default. Exit code 0 means no flagged document for ``scan``, or
successful redaction; 1 means ``scan`` flagged at least one document; 2 means
an input, configuration or backend error. On error, no clean result is
available for the failed document; earlier documents may already have output.

Native categories default to ``secrets,pii,profanity``. Select subsets with
``--categories``, ``--kinds``, ``--exclude-kinds`` and ``--locales`` (comma
lists); use ``--min-confidence`` for the flagging/redaction cutoff and
``--mode placeholder|mask|partial`` for redaction style.

After installing optional dependencies, enable backends explicitly:

.. code-block:: bash

   pygarble-screen scan document.txt --backends phonenumbers,stdnum \
     --backend-options '{"phonenumbers":{"region":"GB"}}'
   pygarble-screen scan document.txt --categories secrets \
     --backends detect-secrets,gitleaks

``--no-builtin`` disables native rules, so at least one selected backend
must supply a detector for the chosen categories. Backend option keys are
documented in :doc:`standalone-screening`. The two CLIs have different flags:
the document CLI has no gibberish ``--profile``, JSON field extraction or
``--format`` setting. Use ``pygarble-screen --help`` for its complete list.

Gibberish line commands
-----------------------

The following ``pygarble`` commands read UTF-8 one text per line; invalid
UTF-8 is replaced with U+FFFD.

Check lines from stdin or files
-------------------------------

.. code-block:: console

   $ printf 'hello world\nasdfghjkl\n' | pygarble check
   clean	hello world
   garbled	asdfghjkl
   $ echo $?
   1

Exit code 0 means nothing was flagged, 1 means at least one line was, 2
means a usage or input error. Blank lines print ``insufficient``. Pass file
names to read files instead of stdin, where ``-`` means stdin, or pass
``-t TEXT`` one or more times to evaluate literal texts.

Scores and full analyses
------------------------

.. code-block:: console

   $ pygarble score -t "please review qxzjkwpvm"
   0.9974	please review qxzjkwpvm
   $ pygarble analyze -t "please review qxzjkwpvm"
   {"text": "please review qxzjkwpvm", "garbled": true, "score": 0.9973843610362856, "status": "garbled", ...}

``analyze`` prints one JSON object per input with the decision, score,
status, profile, spans and per-strategy signals. Long JSON lines are
shortened with ``...`` on this page.

Options shared by ``check``, ``score`` and ``analyze``
------------------------------------------------------

``--profile NAME`` or ``--strategy NAME``, ``--threshold FLOAT``,
``--allowlist FILE`` (one word per line, ``#`` comments), ``--format
text|tsv|jsonl`` and ``--field NAME`` for JSON-lines input, which echoes
each object with a ``pygarble`` key added.

.. code-block:: console

   $ pygarble check --format tsv -t "hello world" -t asdfghjkl
   0	0.0680	clean	hello world
   1	1.0000	garbled	asdfghjkl
   $ printf '{"id":1,"msg":"hello"}\n' | pygarble check --field msg
   {"id": 1, "msg": "hello", "pygarble": {"garbled": false, "score": 0.0566261612139897, "status": "clean", ...}}

The TSV columns are the decision as ``1`` or ``0``, the score, the status
and the text.

Calibrate a threshold
---------------------

``calibrate`` reads one garbled sample per line from ``--garbled`` and one
clean sample per line from ``--clean``, prints precision, recall, F1 and
false-positive rate at every observed score, and recommends a threshold.
``--max-fpr 0.01`` caps the false-positive rate instead of maximising F1;
it implies ``--objective max_fpr`` and is rejected with ``--objective f1``.
Text output starts with the objective in use.

.. code-block:: console

   $ printf 'qxzjkwpv bnmqwer\nasdfghjkl\nzzkqxv wqpt\n' > bad.txt
   $ printf 'hello world\nplease send the invoice\nthe meeting moved to friday\n' > good.txt
   $ pygarble calibrate --garbled bad.txt --clean good.txt
   objective: f1
   threshold	precision	recall	f1	fpr
   0.0000	0.500	1.000	0.667	1.000
   0.0435	0.500	1.000	0.667	1.000
   0.0477	0.600	1.000	0.750	0.667
   0.0680	0.750	1.000	0.857	0.333
   1.0000	1.000	1.000	1.000	0.000
   recommended threshold: 0.5340 (f1=1.000, fpr=0.000)

Pass the recommended value back with ``--threshold``. See
:doc:`calibration` for the Python API.

Compatible combined line scanner
--------------------------------

``pygarble scan`` runs the :class:`~pygarble.Scanner` over each input line
and prints one row per line. ``pygarble redact`` prints the redacted line.
All four categories, including gibberish, are enabled by default for scanning.
These commands do not accept optional backends. Use the document CLI above
for findings-only scan reports or multiline private keys.

.. code-block:: bash

   printf 'mail a@b.co\nhello\n' | pygarble scan
   printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact --mode mask
   pygarble scan --categories secrets,pii --format jsonl records.jsonl
   pygarble redact --field message --mode partial events.jsonl

Options shared by both: ``--categories``, ``--kinds``, ``--exclude-kinds``,
``--locales`` (comma lists), ``--min-confidence``, ``--profile``,
``--threshold`` and ``--allowlist`` for the gibberish category, and
``--field NAME`` to read a JSON object per line. ``--categories``,
``--kinds`` and ``--locales`` need at least one name; ``--exclude-kinds``
may be empty. ``scan`` adds
``--format text|tsv|jsonl`` and ``--show-matches``. Text rows are the label,
the kinds and the line; TSV rows are the decision as ``1`` or ``0``, the
finding count, the kinds and the line. Both count only findings at
``--min-confidence`` or above (and any gibberish finding), so a clean row
names no kind; JSONL rows carry every finding. Without it, matched
substrings are omitted from the findings, but every row still carries the
input line; use ``redact`` when output goes to logs. ``redact`` adds ``--mode
placeholder|mask|partial``, ``--placeholder`` (fields ``{KIND}``, ``{kind}``,
``{category}``) and ``--mask-char``.

Exit codes: ``scan`` returns 1 when any line was flagged; ``redact`` returns
0; both return 2 on bad input or options.
