pygarble Documentation
======================

**Detect gibberish, garbled text, and nonsense with high precision.**

A zero-dependency Python library for identifying random character sequences, keyboard mashing,
encoding errors, and other forms of text corruption.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   strategies
   api
   examples
   contributing

Features
--------

- **24 Detection Strategies**: From Markov chains to phonotactic rules
- **99.5% Precision**: Default ensemble minimizes false positives
- **Zero Dependencies**: Core library uses only Python stdlib
- **Scikit-learn Interface**: Familiar ``predict()`` and ``predict_proba()`` methods
- **Batch Processing**: Process lists of texts efficiently

Quick Start
-----------

.. code-block:: python

   from pygarble import EnsembleDetector

   # Recommended: Use the default ensemble
   detector = EnsembleDetector()

   detector.predict("Hello world")    # False - valid text
   detector.predict("asdfghjkl")      # True - keyboard mashing
   detector.predict("qxzjkwp")        # True - impossible letters

   # Batch processing
   texts = ["Hello world", "asdfghjkl", "Normal text"]
   results = detector.predict(texts)  # [False, True, False]

Performance
-----------

Tested on 1,644 samples:

==================== ========= ====== ========
Detector             Precision Recall F1 Score
==================== ========= ====== ========
EnsembleDetector()   **99.2%** 85.6%  91.9%
MARKOV_CHAIN         99.2%     84.3%  91.2%
LOG_LIKELIHOOD_RATIO 100%      63.4%  77.6%
==================== ========= ====== ========

Installation
------------

.. code-block:: bash

   pip install pygarble

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
