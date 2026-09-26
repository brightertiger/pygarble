Installation
============

pygarble requires Python 3.8 or later. The base install has no runtime
dependencies; its bundled character statistics and dictionary require no
inference-time downloads. Optional screening backends have separate extras.

This documentation includes unreleased changes: ``pygarble.screening``,
``pygarble.gibberish``, optional backends and the document CLI. They are
listed under Unreleased in the repository changelog. Install a checkout of
the reviewed branch or commit to use them; do not assume an existing PyPI
release includes these APIs. See :doc:`migration` for the version boundary.

Install a release
-----------------

.. code-block:: bash

   python -m pip install pygarble
   python -m pip show pygarble

Install from source
-------------------

With Git installed:

.. code-block:: bash

   python -m pip install "pygarble @ git+https://github.com/brightertiger/pygarble.git@main"

Replace ``main`` with a reviewed commit SHA to pin a reproducible installation.
For an editable checkout:

.. code-block:: bash

   git clone https://github.com/brightertiger/pygarble.git
   cd pygarble
   python -m pip install -e .

When evaluating an unmerged change, check out its reviewed branch or commit
before the install command. Once it is merged, ``main`` includes it.

Optional screening backends
---------------------------

From a checkout containing the screening changes:

.. code-block:: bash

   python -m pip install -e ".[phones]"     # phonenumberslite
   python -m pip install -e ".[stdnum]"     # python-stdnum
   python -m pip install -e ".[secrets]"    # detect-secrets
   python -m pip install -e ".[screening]"  # all three Python extras

For a released version containing these changes, the corresponding package
form is ``python -m pip install 'pygarble[screening]'``. Installing an extra
does not enable it; pass backend names to the standalone ``Scanner``.
Native profanity detection uses the bundled English word list and needs no
extra package.

Gitleaks is a separately installed executable, not a Python extra. The
adapter supports Gitleaks >=8.19,<9; CI exercises 8.30.1. Put the executable
on PATH or configure its path through ``backend_options``. The adapter
does not install it. See :doc:`standalone-screening` for supported options
and coverage, and :doc:`cli` for command-line selection.

Verify the source installation
------------------------------

.. code-block:: python

   import pygarble
   from pygarble.gibberish import EnsembleDetector
   from pygarble.screening import Scanner

   print(pygarble.__version__)
   detector = EnsembleDetector()
   assert detector.predict("Hello world") is False
   assert detector.predict("asdfghjkl") is True
   assert Scanner().redact("mail jane@example.com").text == "mail [EMAIL]"

Development tools
-----------------

The ``dev`` extra installs test, formatting, lint, and typing tools. The
``docs`` extra installs Sphinx and the documentation theme.
Neither is required to use the package. ``requirements-dev.txt`` also
includes documentation dependencies for the repository's full CI setup.

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

See :doc:`quickstart` for usage and :doc:`migration` when upgrading.
