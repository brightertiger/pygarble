Quick Start Guide
=================

This guide will get you up and running with pygarble in minutes.

Basic Usage
-----------

The recommended way to use pygarble is with the default ``EnsembleDetector``:

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()

   # Check single text
   detector.predict("Hello world")    # False - valid text
   detector.predict("asdfghjkl")      # True - gibberish

   # Check multiple texts
   texts = ["Hello world", "asdfghjkl", "Normal sentence"]
   results = detector.predict(texts)  # [False, True, False]

   # Get probability scores (0.0 = valid, 1.0 = gibberish)
   detector.predict_proba("Hello world")  # ~0.1
   detector.predict_proba("xkqzjwp")      # ~0.9

Using Individual Strategies
---------------------------

For specific use cases, use individual strategies:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Best overall performance
   detector = GarbleDetector(Strategy.MARKOV_CHAIN)
   detector.predict("hello world")   # False
   detector.predict("xkqzjwpmv")     # True

   # Zero false positives
   detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
   detector.predict("hello world")   # False
   detector.predict("qxjjxz")        # True

   # Detect encoding corruption
   detector = GarbleDetector(Strategy.MOJIBAKE)
   detector.predict("Café")          # False - valid UTF-8
   detector.predict("CafÃ©")         # True - mojibake

   # Detect homoglyph attacks
   detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
   detector.predict("paypal")        # False - all Latin
   detector.predict("pаypal")        # True - Cyrillic 'а'

Custom Ensemble
---------------

Create custom ensembles with specific strategies:

.. code-block:: python

   from pygarble import EnsembleDetector, Strategy

   # Pick your strategies
   detector = EnsembleDetector(
       strategies=[
           Strategy.MARKOV_CHAIN,
           Strategy.BIGRAM_PROBABILITY,
           Strategy.KEYBOARD_PATTERN,
       ]
   )

   # Change voting mode
   detector = EnsembleDetector(voting="any")       # High recall
   detector = EnsembleDetector(voting="all")       # High precision
   detector = EnsembleDetector(voting="majority")  # Balanced (default)

Adjusting Threshold
-------------------

The threshold controls the cutoff for ``predict()``:

.. code-block:: python

   # Lower threshold = more sensitive (more false positives)
   detector = GarbleDetector(Strategy.MARKOV_CHAIN, threshold=0.3)

   # Higher threshold = less sensitive (more false negatives)
   detector = GarbleDetector(Strategy.MARKOV_CHAIN, threshold=0.7)

   # predict_proba() is not affected by threshold
   detector.predict_proba("text")  # Returns 0.0-1.0

Common Patterns
---------------

**Filter user input:**

.. code-block:: python

   detector = EnsembleDetector()

   def validate_input(text):
       if detector.predict(text):
           return "Please enter valid text"
       return None

**Clean a dataset:**

.. code-block:: python

   detector = GarbleDetector(Strategy.MARKOV_CHAIN)
   clean_data = [t for t in raw_data if not detector.predict(t)]

**Detect encoding issues:**

.. code-block:: python

   detector = GarbleDetector(Strategy.MOJIBAKE)
   for text in documents:
       if detector.predict(text):
           print(f"Encoding issue: {text[:50]}")

Next Steps
----------

- Learn about each strategy: :doc:`strategies`
- See practical examples: :doc:`examples`
- Explore the full API: :doc:`api`
