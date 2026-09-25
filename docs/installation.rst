Installation
============

pygarble requires Python 3.8 or later and has no runtime dependencies. Its
bundled character models and dictionary require no inference-time downloads.

Install
-------

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

Verify installation
-------------------

.. code-block:: python

   import pygarble
   from pygarble import EnsembleDetector

   print(pygarble.__version__)
   detector = EnsembleDetector()
   assert detector.predict("Hello world") is False
   assert detector.predict("asdfghjkl") is True

Development tools
-----------------

The ``dev`` extra installs test, formatting, lint, and typing tools. Documentation
build dependencies are included in ``requirements-dev.txt`` in the repository.
Neither is required to use the package.

.. code-block:: bash

   python -m pip install -e ".[dev]"
   python -m pip install -r requirements-dev.txt

See :doc:`quickstart` for usage and :doc:`migration` when upgrading.
