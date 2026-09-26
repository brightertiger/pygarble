Command-line interface
======================

``pygarble`` is installed as a console script; ``python -m pygarble`` is
equivalent. Input is read as UTF-8, one text per line, and invalid UTF-8 is
replaced with U+FFFD.

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

Screening: ``scan`` and ``redact``
----------------------------------

``pygarble scan`` runs the :class:`~pygarble.Scanner` over each input line
and prints one row per line. ``pygarble redact`` prints the redacted line.

.. code-block:: bash

   printf 'mail a@b.co\nhello\n' | pygarble scan
   printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact --mode mask
   pygarble scan --categories secrets,pii --format jsonl records.jsonl
   pygarble redact --field message --mode partial events.jsonl

Options shared by both: ``--categories``, ``--kinds``, ``--exclude-kinds``,
``--locales`` (comma lists), ``--min-confidence``, ``--profile``,
``--threshold`` and ``--allowlist`` for the gibberish category, and
``--field NAME`` to read a JSON object per line. ``scan`` adds
``--format text|tsv|jsonl`` and ``--show-matches``. Without it, matched
substrings are omitted from the findings, but every row still carries the
input line; use ``redact`` when output goes to logs. ``redact`` adds ``--mode
placeholder|mask|partial``, ``--placeholder`` (fields ``{KIND}``, ``{kind}``,
``{category}``) and ``--mask-char``.

Exit codes: ``scan`` returns 1 when any line was flagged; ``redact`` returns
0; both return 2 on bad input or options.
