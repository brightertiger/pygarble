Detection Strategies
====================

pygarble provides 24 strategies for detecting garbled text. Each strategy uses a different
approach and is suited for different types of gibberish.

Recommended Strategies
----------------------

These strategies offer the best balance of precision and recall:

================= ========= ====== ==================================
Strategy          Precision Recall Description
================= ========= ====== ==================================
MARKOV_CHAIN      98.8%     86.4%  Character transition probabilities
NGRAM_FREQUENCY   96.3%     75.8%  Common English trigram analysis
WORD_LOOKUP       92.7%     89.3%  50K English word dictionary
BIGRAM_PROBABILITY 100%     33.6%  Impossible letter pairs
LETTER_POSITION   99.0%     52.9%  Invalid letter positions
================= ========= ====== ==================================

High-Precision Strategies (v0.5.0)
----------------------------------

These strategies are designed to minimize false positives:

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   # Detect impossible letter pairs (qx, jj, xz)
   detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
   detector.predict("qxjjxz")        # True

   # Detect invalid letter positions
   detector = GarbleDetector(Strategy.LETTER_POSITION)
   detector.predict("wordj")         # True - 'j' can't end words

   # Detect too many consonants
   detector = GarbleDetector(Strategy.CONSONANT_SEQUENCE)
   detector.predict("bcdfghjk")      # True - too many consonants

   # Detect invalid vowel patterns
   detector = GarbleDetector(Strategy.VOWEL_PATTERN)
   detector.predict("aaaaaaa")       # True - repeated vowels

   # Detect abnormal letter distribution
   detector = GarbleDetector(Strategy.LETTER_FREQUENCY)
   detector.predict("qqqxxxzzz")     # True - rare letters dominate

   # Detect impossible trigrams
   detector = GarbleDetector(Strategy.RARE_TRIGRAM)
   detector.predict("jjjqqq")        # True

Core Strategies
---------------

**MARKOV_CHAIN** - Best overall performance

Uses character-level Markov chain trained on English text.

.. code-block:: python

   detector = GarbleDetector(Strategy.MARKOV_CHAIN)
   detector.predict("hello world")   # False
   detector.predict("xkqzjwpmv")     # True

**NGRAM_FREQUENCY** - Trigram analysis

Checks what proportion of trigrams appear in common English.

.. code-block:: python

   detector = GarbleDetector(Strategy.NGRAM_FREQUENCY)
   detector.predict("the quick")     # False - common trigrams
   detector.predict("xzqkjh")        # True - no common trigrams

**WORD_LOOKUP** - Dictionary-based

Validates words against embedded 50K English dictionary.

.. code-block:: python

   detector = GarbleDetector(Strategy.WORD_LOOKUP)
   detector.predict("hello world")   # False
   detector.predict("xyzzy plugh")   # True

**PRONOUNCEABILITY** - Phonotactic rules

Checks if text follows English pronunciation rules.

.. code-block:: python

   detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
   detector.predict("strength")      # False - valid clusters
   detector.predict("bvnk tspk")     # True - unpronounceable

Specialized Detectors
---------------------

**MOJIBAKE** - Encoding corruption

Detects UTF-8 text incorrectly decoded as Latin-1.

.. code-block:: python

   detector = GarbleDetector(Strategy.MOJIBAKE)
   detector.predict("Café")          # False - valid UTF-8
   detector.predict("CafÃ©")         # True - mojibake

**UNICODE_SCRIPT** - Homoglyph attacks

Detects Cyrillic/Greek characters disguised as Latin.

.. code-block:: python

   detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
   detector.predict("paypal")        # False - all Latin
   detector.predict("pаypal")        # True - Cyrillic 'а'

**KEYBOARD_PATTERN** - Keyboard sequences

Detects keyboard row patterns (qwerty, asdf).

.. code-block:: python

   detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
   detector.predict("asdfghjkl")     # True
   detector.predict("hello world")   # False

**HEX_STRING** - Hash detection

Detects MD5/SHA hashes and UUIDs.

.. code-block:: python

   detector = GarbleDetector(Strategy.HEX_STRING)
   detector.predict("5d41402abc4b2a76b9719d911017c592")  # True

**SYMBOL_RATIO** - Excessive symbols

Detects text with too many symbols or numbers.

.. code-block:: python

   detector = GarbleDetector(Strategy.SYMBOL_RATIO)
   detector.predict("!!!@@@###")     # True

**REPETITION** - Repeated patterns

Detects repeated characters or patterns.

.. code-block:: python

   detector = GarbleDetector(Strategy.REPETITION)
   detector.predict("ababababab")    # True

All Strategies
--------------

**High Precision**

- ``BIGRAM_PROBABILITY`` - Impossible letter pairs
- ``LETTER_POSITION`` - Invalid letter positions
- ``CONSONANT_SEQUENCE`` - Too many consonants
- ``VOWEL_PATTERN`` - Invalid vowel sequences
- ``LETTER_FREQUENCY`` - Abnormal letter distribution
- ``RARE_TRIGRAM`` - Impossible trigrams

**Core**

- ``MARKOV_CHAIN`` - Character Markov chain
- ``NGRAM_FREQUENCY`` - Trigram frequency
- ``WORD_LOOKUP`` - Dictionary lookup
- ``PRONOUNCEABILITY`` - Phonotactic rules
- ``KEYBOARD_PATTERN`` - Keyboard sequences
- ``ENTROPY_BASED`` - Shannon entropy
- ``VOWEL_RATIO`` - Vowel/consonant ratio

**Specialized**

- ``MOJIBAKE`` - Encoding errors
- ``UNICODE_SCRIPT`` - Homoglyph attacks
- ``HEX_STRING`` - Hash strings
- ``SYMBOL_RATIO`` - Symbol detection
- ``REPETITION`` - Pattern repetition
- ``COMPRESSION_RATIO`` - Compression analysis

**Legacy**

- ``CHARACTER_FREQUENCY`` - Character distribution
- ``WORD_LENGTH`` - Word length analysis
- ``PATTERN_MATCHING`` - Regex patterns
- ``STATISTICAL_ANALYSIS`` - Alphabetic ratio
- ``ENGLISH_WORD_VALIDATION`` - Requires pyspellchecker
