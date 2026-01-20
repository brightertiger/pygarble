API Reference
=============

GarbleDetector
--------------

The main class for single-strategy detection.

.. code-block:: python

   from pygarble import GarbleDetector, Strategy

   GarbleDetector(
       strategy: Strategy,
       threshold: float = 0.5,
       **kwargs
   )

**Parameters:**

- ``strategy``: The detection strategy to use (see Strategy enum)
- ``threshold``: Probability threshold for ``predict()`` (0.0-1.0)
- ``**kwargs``: Strategy-specific parameters

**Methods:**

- ``predict(text)`` - Returns ``bool`` or ``List[bool]``
- ``predict_proba(text)`` - Returns ``float`` or ``List[float]`` (0.0-1.0)

**Example:**

.. code-block:: python

   detector = GarbleDetector(Strategy.MARKOV_CHAIN, threshold=0.5)

   detector.predict("hello")           # False
   detector.predict("xkqzj")           # True
   detector.predict_proba("hello")     # 0.1
   detector.predict(["a", "b", "c"])   # [False, False, False]

EnsembleDetector
----------------

Combines multiple strategies with voting.

.. code-block:: python

   from pygarble import EnsembleDetector, Strategy

   EnsembleDetector(
       strategies: List[Strategy] = None,
       threshold: float = 0.5,
       voting: str = "majority",
       weights: List[float] = None,
   )

**Parameters:**

- ``strategies``: List of strategies (default: high-precision mix)
- ``threshold``: Probability threshold for ``predict()``
- ``voting``: Voting mode - "majority", "any", "all", "average", "weighted"
- ``weights``: Weights for weighted voting (required if voting="weighted")

**Default Strategies:**

- MARKOV_CHAIN
- WORD_LOOKUP
- NGRAM_FREQUENCY
- BIGRAM_PROBABILITY
- LETTER_POSITION

**Voting Modes:**

- ``majority``: Flag if >50% of strategies agree (default)
- ``any``: Flag if ANY strategy detects (high recall)
- ``all``: Flag only if ALL strategies agree (high precision)
- ``average``: Average probability across strategies
- ``weighted``: Weighted average with custom weights

**Example:**

.. code-block:: python

   # Default ensemble
   detector = EnsembleDetector()

   # Custom strategies
   detector = EnsembleDetector(
       strategies=[Strategy.MARKOV_CHAIN, Strategy.KEYBOARD_PATTERN]
   )

   # High recall mode
   detector = EnsembleDetector(voting="any")

Strategy Enum
-------------

Available detection strategies:

**High Precision (v0.5.0)**

- ``BIGRAM_PROBABILITY`` - Impossible letter pairs
- ``LETTER_POSITION`` - Invalid letter positions
- ``CONSONANT_SEQUENCE`` - Too many consonants
- ``VOWEL_PATTERN`` - Invalid vowel sequences
- ``LETTER_FREQUENCY`` - Abnormal letter distribution
- ``RARE_TRIGRAM`` - Impossible trigrams

**Core Strategies**

- ``MARKOV_CHAIN`` - Character Markov chain (recommended)
- ``NGRAM_FREQUENCY`` - Trigram frequency
- ``WORD_LOOKUP`` - 50K English dictionary
- ``PRONOUNCEABILITY`` - Phonotactic rules
- ``KEYBOARD_PATTERN`` - Keyboard sequences
- ``ENTROPY_BASED`` - Shannon entropy
- ``VOWEL_RATIO`` - Vowel/consonant ratio

**Specialized**

- ``MOJIBAKE`` - Encoding corruption
- ``UNICODE_SCRIPT`` - Homoglyph attacks
- ``HEX_STRING`` - Hash strings
- ``SYMBOL_RATIO`` - Excessive symbols
- ``REPETITION`` - Pattern repetition
- ``COMPRESSION_RATIO`` - Compression analysis

**Legacy**

- ``CHARACTER_FREQUENCY``
- ``WORD_LENGTH``
- ``PATTERN_MATCHING``
- ``STATISTICAL_ANALYSIS``
- ``ENGLISH_WORD_VALIDATION`` (requires pyspellchecker)
