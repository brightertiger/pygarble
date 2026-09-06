Contributing
============

pygarble is a deterministic Python package for English gibberish detection.
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
   python -m sphinx -W --keep-going -b html docs /tmp/pygarble-docs

The strategy reference is generated from ``pygarble/registry.py`` and
``pygarble/options.py``. Update those sources when supported settings change, then
run ``python scripts/update_strategy_docs.py``. Keep practical usage guidance in
``docs/strategy-guide.rst`` and runnable examples in the quick start and README.

Data and evaluation
-------------------

.. code-block:: bash

   python scripts/generate_data.py --check
   python regression/evaluate.py --split development

Data generation downloads a pinned source and verifies its checksum. For offline
verification, supply ``--source /path/to/count_1w.txt``. Curated exclusions live in
``scripts/data_curation.json``; artifact hashes live in ``pygarble/data/manifest.json``.
Do not edit generated tables directly.

Keep development and holdout families separate; do not tune thresholds on holdout
errors. Report confusion counts and limitations rather than treating a small
benchmark as a production precision estimate. Hindi being flagged by English
checks is expected; corruption-only checks have a separate contract.

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
