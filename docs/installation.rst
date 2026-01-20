Installation
============

Requirements
------------

- Python 3.8+
- Zero dependencies (core library)

Install from PyPI
-----------------

.. code-block:: bash

   pip install pygarble

Optional Dependencies
---------------------

For the ``ENGLISH_WORD_VALIDATION`` strategy:

.. code-block:: bash

   pip install pygarble[spellchecker]

Verify Installation
-------------------

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()
   print(detector.predict("Hello world"))  # False
   print(detector.predict("asdfghjkl"))    # True

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/brightertiger/pygarble.git
   cd pygarble
   pip install -e ".[dev]"

Run Tests
---------

.. code-block:: bash

   pytest tests/ -v
