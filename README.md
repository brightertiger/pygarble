# pygarble

**Detect gibberish, garbled text, and nonsense with high precision.**

A zero-dependency Python library for identifying random character sequences, keyboard mashing, encoding errors, and other forms of text corruption. Uses statistical analysis, phonotactic rules, and pattern matching to distinguish meaningful text from gibberish.

## Installation

```bash
pip install pygarble
```

## Quick Start

```python
from pygarble import GarbleDetector, EnsembleDetector, Strategy

# Recommended: Use the default ensemble (99.5% precision)
detector = EnsembleDetector()
detector.predict("Hello world")      # False - valid text
detector.predict("asdfghjkl")        # True - keyboard mashing
detector.predict("qxzjkwp")          # True - impossible letter combinations

# Get probability scores (0.0 = valid, 1.0 = gibberish)
detector.predict_proba("Hello world")  # ~0.1
detector.predict_proba("xkqzjwp")      # ~0.9

# Batch processing
texts = ["Hello world", "asdfghjkl", "Normal sentence here"]
results = detector.predict(texts)      # [False, True, False]
```

## Performance

Tested on 1,644 samples (dictionary words, sentences, random strings, keyboard mashing) via `regression/benchmark.py`:

| Detector | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| **EnsembleDetector()** | **100%** | 74.9% | 85.6% |
| MARKOV_CHAIN | 99.2% | 84.3% | 91.2% |
| NGRAM_FREQUENCY | 97.6% | 75.0% | 84.8% |
| LOG_LIKELIHOOD_RATIO | 100% | 63.4% | 77.6% |
| WORD_ANOMALY | 100% | 52.8% | 69.1% |

The default ensemble prioritizes **precision** (minimizing false positives) over recall. For broader coverage of specialist domains (hashes, mojibake, repeated junk, homoglyphs) at slightly more false-positive risk, combine the precise specialists with `voting="any"`:

```python
# High-recall configuration: each member is individually high-precision,
# any single detection flags the text
detector = EnsembleDetector(
    strategies=[
        Strategy.LOG_LIKELIHOOD_RATIO, Strategy.WORD_ANOMALY,
        Strategy.KEYBOARD_ADJACENCY, Strategy.MOJIBAKE,
        Strategy.HEX_STRING, Strategy.REPETITION, Strategy.UNICODE_SCRIPT,
    ],
    voting="any",
)
```

## Detection Strategies

### Recommended Strategies

| Strategy | Description | Precision |
|----------|-------------|-----------|
| `MARKOV_CHAIN` | Character transition probabilities trained on English | 99.2% |
| `NGRAM_FREQUENCY` | Common English trigram analysis | 97.6% |
| `LOG_LIKELIHOOD_RATIO` | English-vs-random two-model bigram comparison | 100% |
| `WORD_ANOMALY` | Per-word scoring; catches one garbage token in a valid sentence | 100% |
| `WORD_LOOKUP` | Dictionary of 49K English words | high recall |

### All Available Strategies

**Statistical Models (v0.7.0)**
- `LOG_LIKELIHOOD_RATIO` - Log-likelihood ratio of English vs uniform character models (length-normalized)
- `WORD_ANOMALY` - Fraction of individually-anomalous words; robust to garbage embedded in valid text
- `KEYBOARD_ADJACENCY` - Physical key-adjacency walks (catches mash the trigram lists miss)

**High Precision (v0.5.0)**
- `BIGRAM_PROBABILITY` - Impossible letter pairs
- `LETTER_POSITION` - Invalid letter positions
- `CONSONANT_SEQUENCE` - Too many consecutive consonants
- `VOWEL_PATTERN` - Invalid vowel sequences
- `LETTER_FREQUENCY` - Abnormal letter distribution
- `RARE_TRIGRAM` - Impossible trigrams

**Core Strategies**
- `MARKOV_CHAIN` - Character-level Markov chain (best overall)
- `NGRAM_FREQUENCY` - Trigram frequency analysis
- `WORD_LOOKUP` - English dictionary lookup
- `PRONOUNCEABILITY` - English phonotactic rules
- `KEYBOARD_PATTERN` - Keyboard row sequences
- `ENTROPY_BASED` - Shannon entropy analysis
- `VOWEL_RATIO` - Vowel to consonant ratio

**Specialized Detectors**
- `MOJIBAKE` - Encoding corruption (UTF-8 as Latin-1)
- `UNICODE_SCRIPT` - Homoglyph/script mixing attacks
- `HEX_STRING` - Hash strings and UUIDs
- `SYMBOL_RATIO` - Excessive symbols/numbers
- `REPETITION` - Repeated patterns (ababab)
- `COMPRESSION_RATIO` - Compression-based detection

**Legacy Strategies**
- `CHARACTER_FREQUENCY`, `WORD_LENGTH`, `PATTERN_MATCHING`, `STATISTICAL_ANALYSIS`
- `ENGLISH_WORD_VALIDATION` (requires `pip install pygarble[spellchecker]`)

## Using Individual Strategies

```python
from pygarble import GarbleDetector, Strategy

# Markov chain - best overall performance
detector = GarbleDetector(Strategy.MARKOV_CHAIN)
detector.predict("the quick brown fox")  # False
detector.predict("xkqzjwpmv")            # True

# High precision - zero false positives
detector = GarbleDetector(Strategy.BIGRAM_PROBABILITY)
detector.predict("hello world")          # False
detector.predict("qxjjxz")               # True (impossible: qx, jj, xz)

# Encoding corruption detection
detector = GarbleDetector(Strategy.MOJIBAKE)
detector.predict("Café")                 # False - valid UTF-8
detector.predict("CafÃ©")                # True - mojibake

# Homoglyph attack detection
detector = GarbleDetector(Strategy.UNICODE_SCRIPT)
detector.predict("paypal")               # False - all Latin
detector.predict("pаypal")               # True - Cyrillic 'а'
```

## Ensemble Detector

Combine multiple strategies for better accuracy:

```python
from pygarble import EnsembleDetector, Strategy

# Default ensemble (recommended)
# Uses: MARKOV_CHAIN, NGRAM_FREQUENCY, WORD_LOOKUP, LOG_LIKELIHOOD_RATIO, WORD_ANOMALY
# Voting: majority
detector = EnsembleDetector()

# Custom strategies
detector = EnsembleDetector(
    strategies=[
        Strategy.MARKOV_CHAIN,
        Strategy.BIGRAM_PROBABILITY,
        Strategy.KEYBOARD_PATTERN,
    ]
)

# Different voting modes
detector = EnsembleDetector(voting="any")       # High recall - flag if ANY strategy detects
detector = EnsembleDetector(voting="all")       # High precision - flag only if ALL agree
detector = EnsembleDetector(voting="majority")  # Balanced (default)
detector = EnsembleDetector(voting="average")   # Average probabilities

# Weighted voting
detector = EnsembleDetector(
    strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_LOOKUP],
    voting="weighted",
    weights=[0.7, 0.3]
)
```

## API Reference

### GarbleDetector

```python
GarbleDetector(
    strategy: Strategy,
    threshold: float = 0.5,    # Probability threshold for predict()
    **kwargs                   # Strategy-specific parameters
)

# Methods
detector.predict(text)         # Returns bool or List[bool]
detector.predict_proba(text)   # Returns float or List[float] (0.0-1.0)
```

### EnsembleDetector

```python
EnsembleDetector(
    strategies: List[Strategy] = None,  # Default: high-precision mix
    threshold: float = 0.5,
    voting: str = "majority",           # "majority", "any", "all", "average", "weighted"
    weights: List[float] = None,        # Required if voting="weighted"
)

# Methods (same as GarbleDetector)
detector.predict(text)
detector.predict_proba(text)
```

## Common Use Cases

### Filter User Input
```python
detector = EnsembleDetector()

def validate_input(text):
    if detector.predict(text):
        return "Please enter valid text"
    return None
```

### Clean Data Pipeline
```python
detector = GarbleDetector(Strategy.MARKOV_CHAIN)

clean_data = [text for text in raw_data if not detector.predict(text)]
```

### Detect Encoding Issues
```python
detector = GarbleDetector(Strategy.MOJIBAKE)

for text in documents:
    if detector.predict(text):
        print(f"Encoding issue detected: {text[:50]}...")
```

### Detect Phishing/Homoglyphs
```python
detector = GarbleDetector(Strategy.UNICODE_SCRIPT)

if detector.predict(domain_name):
    print("Warning: Possible homoglyph attack")
```

## Requirements

- Python 3.8+
- Zero dependencies (core library)
- Optional: `pyspellchecker` for `ENGLISH_WORD_VALIDATION` strategy

## Development

```bash
git clone https://github.com/brightertiger/pygarble.git
cd pygarble
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License

## Changelog

### 0.5.0
- 6 new high-precision strategies (BIGRAM_PROBABILITY, LETTER_POSITION, CONSONANT_SEQUENCE, VOWEL_PATTERN, LETTER_FREQUENCY, RARE_TRIGRAM)
- Redesigned default ensemble for 99.5% precision
- External validation benchmark (1,644 test cases)

### 0.4.0
- Added COMPRESSION_RATIO, MOJIBAKE, PRONOUNCEABILITY, UNICODE_SCRIPT strategies

### 0.3.0
- Zero-dependency core with embedded training data
- Added MARKOV_CHAIN, NGRAM_FREQUENCY, WORD_LOOKUP strategies
