Examples
========

Practical examples for common use cases.

Filter User Input
-----------------

Validate form input before processing:

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()

   def validate_input(text):
       if not text or len(text.strip()) == 0:
           return "Input cannot be empty"
       if detector.predict(text):
           return "Please enter valid text"
       return None

   # Usage
   error = validate_input("Hello world")    # None - valid
   error = validate_input("asdfghjkl")      # "Please enter valid text"

Clean a Dataset
---------------

Remove gibberish from a list of texts:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   detector = GarbleDetector(Strategy.MARKOV_CHAIN)

   raw_data = [
       "This is valid text",
       "asdfghjkl",
       "Another good sentence",
       "qxzjkwpmv",
       "Final valid text"
   ]

   clean_data = [t for t in raw_data if not detector.predict(t)]
   print(clean_data)
   # ['This is valid text', 'Another good sentence', 'Final valid text']

Detect Encoding Issues
----------------------

Find mojibake (encoding corruption) in documents:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   detector = GarbleDetector(Strategy.MOJIBAKE)

   documents = [
       "Café au lait",           # Valid UTF-8
       "CafÃ© au lait",          # Mojibake
       "naïve résumé",           # Valid UTF-8
       "naÃ¯ve rÃ©sumÃ©",       # Mojibake
   ]

   for doc in documents:
       if detector.predict(doc):
           print(f"Encoding issue: {doc}")

Detect Phishing/Homoglyphs
--------------------------

Identify lookalike characters in domain names:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   detector = GarbleDetector(Strategy.UNICODE_SCRIPT)

   domains = [
       "paypal.com",      # Legitimate
       "pаypal.com",      # Cyrillic 'а'
       "google.com",      # Legitimate
       "gооgle.com",      # Cyrillic 'о'
       "apple.com",       # Legitimate
       "аpple.com",       # Cyrillic 'а'
   ]

   for domain in domains:
       if detector.predict(domain):
           print(f"Warning: Possible phishing - {domain}")

Batch Processing
----------------

Process large datasets efficiently:

.. code-block:: python

   from pygarble import EnsembleDetector

   detector = EnsembleDetector()

   # Process 10,000 texts at once
   texts = ["sample text"] * 10000
   results = detector.predict(texts)

   garbled_count = sum(results)
   print(f"Found {garbled_count} gibberish texts")

Custom Strategy Selection
-------------------------

Choose strategies based on your use case:

.. code-block:: python

   from pygarble import EnsembleDetector, Strategy

   # High precision (minimize false positives)
   detector = EnsembleDetector(
       strategies=[
           Strategy.BIGRAM_PROBABILITY,
           Strategy.LETTER_POSITION,
           Strategy.RARE_TRIGRAM,
       ],
       voting="all"  # Only flag if ALL agree
   )

   # High recall (catch everything)
   detector = EnsembleDetector(
       strategies=[
           Strategy.MARKOV_CHAIN,
           Strategy.KEYBOARD_PATTERN,
           Strategy.WORD_LOOKUP,
       ],
       voting="any"  # Flag if ANY detects
   )

Threshold Tuning
----------------

Adjust sensitivity for your needs:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Get probability scores first
   detector = GarbleDetector(Strategy.MARKOV_CHAIN)

   test_texts = [
       ("Hello world", False),      # Clearly valid
       ("xkqzjwpmv", True),         # Clearly gibberish
       ("asdfgh", True),            # Borderline
   ]

   print("Probability scores:")
   for text, _ in test_texts:
       prob = detector.predict_proba(text)
       print(f"  {text:20} -> {prob:.3f}")

   # Then choose appropriate threshold
   # Lower = more sensitive (more false positives)
   # Higher = less sensitive (more false negatives)

Streaming/Real-time Detection
-----------------------------

Process text as it arrives:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Use a fast strategy
   detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)

   def process_message(message):
       if detector.predict(message):
           return {"status": "rejected", "reason": "invalid text"}
       return {"status": "accepted", "message": message}

   # Process incoming messages
   messages = ["Hello", "xqzjk", "World"]
   for msg in messages:
       result = process_message(msg)
       print(f"{msg}: {result['status']}")
