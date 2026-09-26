Contributing
============

pygarble is a deterministic Python package for text screening: secrets, PII,
profanity and English gibberish.
Keep runtime dependencies at zero, document heuristic limitations, and include
regression cases for behavior changes.

Set up a checkout
-----------------

Clone your fork, or use the upstream repository if you have write access:

.. code-block:: bash

   git clone https://github.com/brightertiger/pygarble.git
   cd pygarble
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]"
   python -m pip install -r requirements-dev.txt
   git switch -c docs/my-change

On Windows, activate with ``.venv\Scripts\activate`` instead.

Validation
----------

Run the checks relevant to your change. The CI test matrix covers Python 3.8–3.12.

.. code-block:: bash

   python -m pytest -q
   black --check pygarble tests scripts regression
   isort --check-only pygarble tests scripts regression
   flake8 pygarble tests scripts regression
   mypy pygarble
   pre-commit run --all-files
   python scripts/update_strategy_docs.py --check
   python -m sphinx -n -W --keep-going -b html docs /tmp/pygarble-docs

The strategy reference is generated from ``pygarble/gibberish/registry.py`` and
``pygarble/gibberish/options.py``. Update those sources when supported settings
change, then run ``python scripts/update_strategy_docs.py``. Keep practical usage guidance in
``docs/strategy-guide.rst`` and runnable examples in the quick start and README.

Module ownership and compatibility
----------------------------------

Make gibberish changes under ``pygarble/gibberish/`` and screening changes
under ``pygarble/screening/``. Keep category-specific rules beside their
detector; shared findings, redaction, validation and data stay at the root.
See :doc:`architecture` for the source tree and import mappings.

The old root modules and category packages are compatibility pointers.
Preserve class/enum identity, mutable module state and lazy imports when
changing them. ``tests/test_module_layout.py`` exercises imports in both
orders, patching through old paths and pickles created before the move.
``tests/test_screening.py`` checks that standalone screening does not load
the gibberish engine or optional dependencies by default.

For backend changes, install the Python extras and Gitleaks, then run:

.. code-block:: bash

   python -m pip install -e ".[screening,dev]"
   gitleaks version
   python -m pytest tests/test_screening*.py -q

Set ``PYGARBLE_GITLEAKS`` to an executable path for integration tests if
Gitleaks is not on PATH. Tests requiring unavailable optional tools skip;
report those skips rather than treating them as executed checks. CI tests
all backends on Python 3.8 and 3.12, including a pinned Gitleaks executable.
Keep backend imports optional and do not add inference-time network calls.

For documentation changes, build with Sphinx warnings treated as errors,
check local links, and execute the affected Python examples. Optional-backend
examples need their dependencies installed. Mark unreleased features clearly
and keep the standalone and combined scanner contracts distinct.
The base suite executes every block labelled ``python`` via
``tests/test_docs_snippets.py``. Follow the existing ``text`` block convention
for examples requiring optional packages or external executables, and execute
those examples separately with their dependencies installed.

Data and evaluation
-------------------

Use :doc:`benchmarks` for the paper-based research benchmark.
``make benchmark-check`` runs its fast offline checks; full inference is
optional and writes new outputs rather than replacing paper results.


.. code-block:: bash

   python scripts/generate_data.py --check
   python regression/evaluate.py --split development

Data generation downloads a pinned source and verifies its checksum. For offline
verification, supply ``--source /path/to/count_1w.txt``. Curated exclusions live in
``scripts/data_curation.json``; artifact hashes live in ``pygarble/data/manifest.json``.
Do not edit generated tables directly. The generator also writes
language-neutral JSON copies of the tables (``words.json``, ``bigrams.json``,
``trigrams.json``) for ports to other languages; they are hashed in
``manifest.json`` and verified by ``--check``.

Keep development and holdout families separate; do not tune thresholds on holdout
errors. Report confusion counts and limitations rather than treating a small
benchmark as a production precision estimate. Hindi being flagged by English
checks is expected; corruption-only checks have a separate contract.

The golden corpus ``regression/golden.jsonl`` freezes the detector output
(``garbled``, ``score``, ``status`` and spans) for every challenge case and a
set of edge inputs under every profile. CI runs
``python regression/golden.py --check``; regenerate with
``python regression/golden.py --write`` only when a behaviour change is
intended, and review the diff. Any port must reproduce it exactly; span offsets
are Unicode code points.

Screening rules
---------------

Rule tables live in ``pygarble/screening/secrets/patterns.py``,
``pygarble/screening/pii/patterns.py`` and
``pygarble/screening/profanity/wordlist.py``. Their
JSON copies for ports are regenerated by ``python scripts/generate_data.py``
and verified by ``--check``. Vectors in the JSON table are split into
8-character chunks so secret scanners do not flag them. When you add a rule, extend the matching
precheck and anchor tables too; a length guard raises at import otherwise.
Every new rule needs positive and negative cases in
``regression/scan_vectors.json``; a new PII rule must add at least one
positive vector, so the golden scan corpus covers it.

.. code-block:: bash

   python regression/golden_scan.py --check
   python regression/throughput.py
   python regression/throughput.py --chunk-bytes 4096

``regression/scan_vectors.json`` holds the expected findings for every rule
and ``regression/clean_corpus/`` holds ordinary prose that must stay free of
findings; the test suite checks both. ``golden_scan.py --check`` runs in CI
and compares full :class:`pygarble.Scanner` output over the vectors, the
clean corpus and the gibberish golden inputs with
``regression/golden_scan.jsonl``, which stores text hashes, never text.
Regenerate it with ``--write`` only for an intended behaviour change. The throughput
script reports MB/s per category on a synthetic corpus; use it to compare
before and after a change on the same machine.

Package validation
------------------

.. code-block:: bash

   python -m pip install build twine
   python -m build
   python -m twine check dist/*

Also verify the wheel in a fresh environment with ``pip install --no-deps``.
Building artifacts does not publish them. Publishing is a separate maintainer
action; do not create release tags or dispatch publishing workflows as part of
routine documentation or implementation changes.

Pull requests and issue reports
-------------------------------

Use a feature or fix branch and open a pull request with the problem, resulting
behavior, and relevant validation. Update user documentation when the API or
classification policy changes. Include tests for new behavior or fixes; avoid
claiming unexecuted checks passed.

For an issue, include a minimal input, expected and actual results, Python and
package versions, profile or strategy, and configuration. See existing tests for
examples of batch contracts, explanations, and English-specific behavior.
