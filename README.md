# pygarble

**Detect gibberish, garbled text, and corrupted content with high accuracy using advanced machine learning techniques.**

pygarble is a powerful Python library designed to identify nonsensical, garbled, or corrupted text content that often appears in data processing pipelines, user inputs, or automated systems. Whether you're dealing with random character sequences, encoding errors, keyboard mashing, or corrupted data streams, pygarble provides multiple detection strategies to filter out unwanted content and maintain data quality. The library uses statistical analysis, entropy calculations, pattern matching, and n-gram analysis to distinguish between meaningful text and gibberish with configurable sensitivity levels.

## Features

- **14 Detection Strategies**: Choose from multiple garble detection algorithms including Markov chains and n-gram analysis
- **Zero Dependencies**: Core library works without any external dependencies
- **Ensemble Detector**: Combine multiple strategies for higher accuracy with voting mechanisms
- **Scikit-learn Interface**: Familiar `predict()` and `predict_proba()` methods
- **Configurable Thresholds**: Adjust sensitivity for each strategy
- **Probability Scores**: Get confidence scores for garble detection
- **Input Validation**: Built-in validation for thresholds and parameters
- **Type Hints**: Full type annotation support throughout the codebase
- **Modular Design**: Easy to extend with new detection strategies
- **Smart Edge Cases**: Automatically detects extremely long strings without whitespace (like base64 data)

## Installation

You can install pygarble using pip:

```bash
# Core library (zero dependencies)
pip install pygarble

# With pyspellchecker for legacy word validation (optional)
pip install pygarble[spellchecker]
```

## Quick Start

```python
from pygarble import GarbleDetector, Strategy, EnsembleDetector

# RECOMMENDED: Ensemble with "any" voting (best F1 score: 77.42%)
ensemble = EnsembleDetector(voting="any")
print(ensemble.predict("hello world"))      # False
print(ensemble.predict("asdfghjkl"))        # True
print(ensemble.predict("!!!@@@###"))        # True (symbols detected)
print(ensemble.predict("5d41402abc4b2a76")) # True (hex string detected)

# Individual strategies for specific use cases
detector = GarbleDetector(Strategy.MARKOV_CHAIN)  # High precision (92%)
print(detector.predict("the quick brown fox"))    # False

detector = GarbleDetector(Strategy.WORD_LOOKUP)   # Zero dependencies
print(detector.predict("xyzzy plugh"))            # True

# Batch processing
texts = ["Hello world", "asdfghjkl", "qwertyuiop"]
results = ensemble.predict(texts)
print(results)  # [False, True, True]

# Get probability scores
probabilities = ensemble.predict_proba(texts)
print(probabilities)  # [0.0, 1.0, 0.7]
```

## Benchmark Results

Based on 200 real-world test cases across 34 categories:

### Ensemble Performance (Recommended)

| Configuration | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| **Ensemble (any voting)** ⭐ | **82.50%** | 72.29% | **83.33%** | **77.42%** |
| High Recall (9 strategies) | 75.50% | 60.95% | 88.89% | 72.32% |
| Ensemble (majority voting) | 76.00% | 92.86% | 36.11% | 52.00% |

### Individual Strategy Performance

| Strategy | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| **markov_chain** | 80.00% | **92.11%** | 48.61% | 63.64% |
| **keyboard_pattern** | 78.00% | 70.00% | 68.06% | 69.01% |
| **ngram_frequency** | 78.00% | 83.33% | 48.61% | 61.40% |
| word_lookup | 79.00% | 91.67% | 45.83% | 61.11% |
| vowel_ratio | 78.00% | 85.00% | 47.22% | 60.71% |
| symbol_ratio | 65.50% | 56.52% | 18.06% | 27.37% |
| hex_string | 67.00% | 71.43% | 13.89% | 23.26% |
| repetition | 66.00% | 70.00% | 9.72% | 17.07% |

**Key insights:**
- **Ensemble with "any" voting** achieves the best F1 (77.42%) with 83% recall
- For **high recall** (catching more garbled text), use `voting="any"`
- For **high precision** (avoiding false positives), use individual strategies like `markov_chain` (92% precision)
- New strategies (`symbol_ratio`, `hex_string`, `repetition`) are specialized detectors that complement the core strategies

Run the benchmark yourself:
```bash
python regression/benchmark.py
```

## Detection Strategies

Each strategy implements a different approach to detect garbled text. All strategies return probability scores between 0.0 and 1.0, where higher scores indicate more likely garbled text.

### 1. Keyboard Pattern (`KEYBOARD_PATTERN`) ⭐ Best F1 Score

**Implementation Logic**: Detects keyboard row sequences (qwerty, asdf, zxcv) and analyzes trigram patterns. English text has predictable trigram distributions; garbled text doesn't.

**Algorithm**:
1. Extract trigrams from the text
2. Check for keyboard row sequences (forward and reverse)
3. Compare against common English trigrams
4. Detect repeated bigram patterns (ababab)

**Parameters**:
- `keyboard_threshold` (float, default: 0.3): Threshold for keyboard pattern ratio
- `common_trigram_threshold` (float, default: 0.1): Minimum common trigram ratio

```python
detector = GarbleDetector(Strategy.KEYBOARD_PATTERN, threshold=0.5)

# Examples
detector.predict("asdfghjkl")       # True - keyboard row pattern
detector.predict("qwertyuiop")      # True - keyboard row pattern
detector.predict("Hello world")     # False - normal English text
detector.predict("ababababab")      # True - repeated bigram pattern
```

### 2. Vowel Ratio (`VOWEL_RATIO`) ⭐ Best Precision

**Implementation Logic**: Analyzes the ratio of vowels to consonants. Natural English has 35-45% vowels. Also detects consonant clusters that are impossible in English.

**Algorithm**:
1. Calculate vowel ratio in alphabetic characters
2. Detect long consonant clusters (4+ consecutive consonants)
3. Flag text outside normal vowel ratio range (15-65%)

**Parameters**:
- `min_vowel_ratio` (float, default: 0.15): Minimum allowed vowel ratio
- `max_vowel_ratio` (float, default: 0.65): Maximum allowed vowel ratio
- `consonant_cluster_len` (int, default: 4): Max consonant cluster length

```python
detector = GarbleDetector(Strategy.VOWEL_RATIO, threshold=0.5)

# Examples
detector.predict("bcdfghjklmnp")    # True - no vowels
detector.predict("aeiouaeiou")      # True - all vowels
detector.predict("Hello world")     # False - normal vowel ratio (~36%)
detector.predict("rhythm")          # False - valid English word
```

### 3. Entropy Based (`ENTROPY_BASED`)

**Implementation Logic**: Uses Shannon entropy combined with bigram frequency analysis. English text has predictable character and bigram distributions.

**Algorithm**:
1. Calculate Shannon entropy of alphabetic characters
2. Analyze common English bigram frequency (th, he, in, er, etc.)
3. Combine entropy and bigram scores

**Parameters**:
- `entropy_threshold` (float, default: 2.5): Minimum required entropy
- `bigram_threshold` (float, default: 0.15): Minimum common bigram ratio

```python
detector = GarbleDetector(Strategy.ENTROPY_BASED, threshold=0.5)

# Examples
detector.predict("aaaaaaa")         # True - low entropy (repetitive)
detector.predict("xkjqzpv")         # True - no common bigrams
detector.predict("the weather")     # False - high common bigram ratio
```

### 4. Pattern Matching (`PATTERN_MATCHING`)

**Implementation Logic**: Uses regex patterns to detect suspicious sequences including keyboard rows, repeated characters, and consonant clusters.

**Default Patterns**:
- `special_chars`: 3+ special characters
- `repeated_chars`: 4+ repeated characters
- `uppercase_sequence`: 5+ uppercase letters
- `long_numbers`: 8+ consecutive digits
- `keyboard_row_qwerty`: Keyboard row sequences (qwert, asdf, zxcv)
- `keyboard_row_reverse`: Reverse keyboard sequences
- `consonant_cluster`: 5+ consecutive consonants
- `alternating_pattern`: Alternating character patterns (ababab)

```python
detector = GarbleDetector(Strategy.PATTERN_MATCHING, threshold=0.2)

# Examples
detector.predict("asdfghjkl")       # True - keyboard row
detector.predict("AAAAA")           # True - repeated chars
detector.predict("normal text")     # False - no patterns match
```

### 5. Markov Chain (`MARKOV_CHAIN`) ⭐ NEW - Recommended

**Implementation Logic**: Uses a character-level Markov chain trained on English text. Computes the probability of text based on character transition frequencies. Garbled text has unusual character transitions.

**Algorithm**:
1. Train bigram transition probabilities on 300K+ English words
2. Compute average log-probability of character transitions
3. Map to garble score using sigmoid function

**Parameters**:
- `threshold_per_char` (float, default: -3.5): Average log probability threshold

```python
detector = GarbleDetector(Strategy.MARKOV_CHAIN, threshold=0.5)

# Examples
detector.predict("hello world")       # False - common bigrams (he, el, ll, lo, ow, wo, or, rl, ld)
detector.predict("asdfghjkl")         # True - unusual bigrams (sd, df, fg, gh, hj, jk, kl)
detector.predict("xzqkjhf")           # True - rare character transitions
```

### 6. N-gram Frequency (`NGRAM_FREQUENCY`) ⭐ NEW

**Implementation Logic**: Analyzes what proportion of character trigrams appear in common English text. Uses a set of 2000 most common English trigrams.

**Algorithm**:
1. Extract trigrams from words
2. Count how many appear in common trigram set
3. Low ratio = likely garbled

**Parameters**:
- `common_ratio_threshold` (float, default: 0.3): Minimum ratio of common trigrams

```python
detector = GarbleDetector(Strategy.NGRAM_FREQUENCY, threshold=0.5)

# Examples
detector.predict("the quick brown")   # False - trigrams: the, qui, uic, ick, bro, row, own
detector.predict("xzqkjhf")           # True - no common trigrams
```

### 7. Word Lookup (`WORD_LOOKUP`) ⭐ NEW - Zero Dependencies

**Implementation Logic**: Validates words against an embedded dictionary of 50,000 common English words. No external dependencies required.

**Algorithm**:
1. Tokenize text into words
2. Check each word against embedded word set
3. Return ratio of unknown words

**Parameters**:
- `unknown_threshold` (float, default: 0.5): Ratio above which text is garbled

```python
detector = GarbleDetector(Strategy.WORD_LOOKUP, threshold=0.5)

# Examples
detector.predict("hello world")       # False - both words in dictionary
detector.predict("xyzzy plugh")       # True - neither word in dictionary
detector.predict("hello xyzzy")       # True (0.5) - half unknown
```

### 8. Symbol Ratio (`SYMBOL_RATIO`) - NEW

**Implementation Logic**: Detects text with high proportion of special characters, numbers, or non-alphabetic content. Particularly effective for symbol spam, number sequences, and mixed alphanumeric noise.

**Algorithm**:
1. Count non-alphabetic characters (excluding spaces)
2. Calculate ratio to total characters
3. High ratio = likely garbled

**Parameters**:
- `symbol_threshold` (float, default: 0.5): Ratio above which text is garbled
- `min_length` (int, default: 3): Minimum text length to analyze
- `allow_spaces` (bool, default: True): Whether to exclude spaces from ratio

```python
detector = GarbleDetector(Strategy.SYMBOL_RATIO, threshold=0.5)

# Examples
detector.predict("!!!@@@###$$$")     # True - all symbols
detector.predict("abc123def456")     # True - high number ratio
detector.predict("hello world")       # False - mostly alphabetic
```

### 9. Repetition (`REPETITION`) - NEW

**Implementation Logic**: Detects text with excessive character or pattern repetition. Identifies repeated single characters, bigrams, trigrams, and low character diversity.

**Algorithm**:
1. Check for repeated single characters (aaaa)
2. Check for repeated bigrams (ababab)
3. Check for repeated trigrams (abcabcabc)
4. Analyze character diversity

**Parameters**:
- `max_char_repeat` (int, default: 3): Maximum allowed consecutive repeated characters
- `max_pattern_repeat` (int, default: 3): Maximum allowed pattern repetitions
- `diversity_threshold` (float, default: 0.3): Minimum unique character ratio

```python
detector = GarbleDetector(Strategy.REPETITION, threshold=0.5)

# Examples
detector.predict("aaaaaaaaaa")        # True - repeated character
detector.predict("abababababab")      # True - repeated bigram
detector.predict("hello world")       # False - diverse characters
```

### 10. Hex String (`HEX_STRING`) - NEW

**Implementation Logic**: Detects hash strings, UUIDs, base64-like content, and other hexadecimal patterns commonly found in garbled data.

**Algorithm**:
1. Check for pure hash patterns (MD5, SHA256)
2. Detect UUID format (8-4-4-4-12)
3. Identify long hex sequences
4. Check for base64-like patterns

**Parameters**:
- `min_hex_length` (int, default: 16): Minimum hex sequence length to detect
- `hex_ratio_threshold` (float, default: 0.7): Ratio of hex chars above which text is suspicious

```python
detector = GarbleDetector(Strategy.HEX_STRING, threshold=0.5)

# Examples
detector.predict("5d41402abc4b2a76b9719d911017c592")  # True - MD5 hash
detector.predict("550e8400-e29b-41d4-a716-446655440000")  # True - UUID
detector.predict("hello world")                        # False - no hex patterns
```

### 11. English Word Validation (`ENGLISH_WORD_VALIDATION`)

**Implementation Logic**: Validates words against an English dictionary using pyspellchecker.

> **Note**: Requires optional dependency. Install with `pip install pygarble[spellchecker]`
> Consider using `WORD_LOOKUP` instead for zero-dependency operation.

```python
detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.5)

# Examples
detector.predict("hello world")              # False - valid words
detector.predict("asdfghjkl qwertyuiop")    # True - invalid words
```

### 12. Character Frequency (`CHARACTER_FREQUENCY`)

**Implementation Logic**: Analyzes character frequency distribution. Garbled text often has skewed distributions.

```python
detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.5)

# Examples
detector.predict("aaaaaaa")         # True - high 'a' frequency
detector.predict("normal text")     # False - balanced distribution
```

### 13. Statistical Analysis (`STATISTICAL_ANALYSIS`)

**Implementation Logic**: Analyzes the ratio of alphabetic to non-alphabetic characters.

```python
detector = GarbleDetector(Strategy.STATISTICAL_ANALYSIS, threshold=0.5)

# Examples
detector.predict("123456789")       # True - no alphabetic chars
detector.predict("normal text")     # False - mostly alphabetic
```

### 14. Word Length (`WORD_LENGTH`)

**Implementation Logic**: Checks average word length against normal English patterns.

```python
detector = GarbleDetector(Strategy.WORD_LENGTH, threshold=0.5)

# Examples
detector.predict("supercalifragilistic")  # True - very long word
detector.predict("short words here")       # False - normal lengths
```

## Ensemble Detector

Combine multiple strategies for better accuracy using voting:

```python
from pygarble import EnsembleDetector, Strategy

# Default ensemble (uses best-performing strategies)
ensemble = EnsembleDetector()
print(ensemble.predict("asdfghjkl"))  # True

# Custom strategies
ensemble = EnsembleDetector(
    strategies=[
        Strategy.KEYBOARD_PATTERN,
        Strategy.VOWEL_RATIO,
        Strategy.ENTROPY_BASED,
    ],
    voting="majority"  # or "average" or "weighted"
)

# Weighted voting
ensemble = EnsembleDetector(
    strategies=[Strategy.KEYBOARD_PATTERN, Strategy.VOWEL_RATIO],
    voting="weighted",
    weights=[0.7, 0.3]
)

# Batch processing
texts = ["Hello world", "asdfghjkl", "qwertyuiop"]
results = ensemble.predict(texts)
probas = ensemble.predict_proba(texts)
```

**Voting Modes**:
- `majority`: Text is garbled if >50% of strategies agree
- `average`: Average probability across all strategies
- `weighted`: Weighted average using custom weights
- `any`: High recall - text is garbled if ANY strategy flags it (best F1 score)
- `all`: High precision - text is garbled only if ALL strategies agree

## Advanced Usage

### Input Validation

The library validates inputs automatically:

```python
# Threshold must be between 0 and 1
detector = GarbleDetector(Strategy.KEYBOARD_PATTERN, threshold=1.5)
# Raises: ValueError: threshold must be between 0.0 and 1.0

# Threads must be positive
detector = GarbleDetector(Strategy.KEYBOARD_PATTERN, threads=0)
# Raises: ValueError: threads must be a positive integer
```

### Batch Processing with Threading

```python
detector = GarbleDetector(Strategy.KEYBOARD_PATTERN, threads=4)

# Process 1000 texts in parallel
texts = ["text"] * 1000
results = detector.predict(texts)
```

### Custom Pattern Matching

```python
custom_patterns = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'\d{3}-\d{3}-\d{4}',
}

detector = GarbleDetector(
    Strategy.PATTERN_MATCHING,
    patterns=custom_patterns,
    override_defaults=True  # Use only custom patterns
)
```

## API Reference

### GarbleDetector

```python
GarbleDetector(
    strategy: Strategy,
    threshold: float = 0.5,
    threads: Optional[int] = None,
    **kwargs
)
```

**Parameters:**
- `strategy`: Detection strategy to use
- `threshold`: Probability threshold (0.0-1.0) for binary predictions
- `threads`: Number of threads for batch processing
- `**kwargs`: Strategy-specific parameters

**Methods:**
- `predict(X)`: Returns `bool` or `List[bool]`
- `predict_proba(X)`: Returns `float` or `List[float]`

### EnsembleDetector

```python
EnsembleDetector(
    strategies: Optional[List[Strategy]] = None,
    threshold: float = 0.5,
    voting: str = "majority",  # "majority", "average", "weighted", "any", "all"
    weights: Optional[List[float]] = None,
    threads: Optional[int] = None,
    **kwargs
)
```

### Strategy Enum

```python
class Strategy(Enum):
    # Core strategies (zero dependencies)
    MARKOV_CHAIN = "markov_chain"              # Recommended - High precision
    NGRAM_FREQUENCY = "ngram_frequency"        # Trigram analysis
    WORD_LOOKUP = "word_lookup"                # Zero dependencies dictionary
    SYMBOL_RATIO = "symbol_ratio"              # Symbol/number detection
    REPETITION = "repetition"                  # Pattern repetition
    HEX_STRING = "hex_string"                  # Hash/UUID detection
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    VOWEL_RATIO = "vowel_ratio"
    KEYBOARD_PATTERN = "keyboard_pattern"

    # Strategy with optional dependency
    ENGLISH_WORD_VALIDATION = "english_word_validation"  # Requires: pygarble[spellchecker]
```

## Architecture

```
pygarble/
├── __init__.py
├── core.py                      # GarbleDetector & EnsembleDetector
├── data/                        # Embedded training data
│   ├── words.py                 # 50K English words
│   ├── bigrams.py               # Character transition probabilities
│   └── trigrams.py              # Common English trigrams
└── strategies/
    ├── base.py                  # BaseStrategy with shared utilities
    ├── markov_chain.py          # Markov chain detection
    ├── ngram_frequency.py       # Trigram frequency analysis
    ├── word_lookup.py           # Dictionary lookup (zero deps)
    ├── symbol_ratio.py          # Symbol/number detection
    ├── repetition.py            # Pattern repetition
    ├── hex_string.py            # Hash/UUID detection
    ├── character_frequency.py
    ├── word_length.py
    ├── pattern_matching.py      # Regex patterns + keyboard detection
    ├── statistical_analysis.py
    ├── entropy_based.py         # Shannon entropy + bigram analysis
    ├── english_word_validation.py  # pyspellchecker (optional)
    ├── vowel_ratio.py           # Vowel analysis + consonant clusters
    └── keyboard_pattern.py      # N-gram + keyboard row detection
```

## Dependencies

**Core library**: Zero dependencies - works with Python 3.8+ only

**Optional dependency**:
- `pygarble[spellchecker]`: pyspellchecker for English word validation
  - pyspellchecker>=0.7.0

## Development

```bash
# Clone and setup
git clone https://github.com/brightertiger/pygarble.git
cd pygarble
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run benchmark
python regression/benchmark.py

# Linting
flake8 pygarble/
black pygarble/
mypy pygarble/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Adding New Strategies

1. Create a new file in `pygarble/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `_predict_impl()` and `_predict_proba_impl()`
4. Add to `strategies/__init__.py` and `core.py`
5. Add tests in `tests/`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

### 0.3.1 (Current)
- **New Strategies**: Added `SYMBOL_RATIO`, `REPETITION`, and `HEX_STRING` strategies for specialized detection
- **New Voting Modes**: Added `any` (high recall) and `all` (high precision) voting modes for EnsembleDetector
- **Removed**: `LANGUAGE_DETECTION` strategy (FastText dependency had NumPy 2.0 compatibility issues)
- **Production-Grade Robustness**:
  - Thread safety with timeout and exception handling in batch processing
  - Division by zero protection across all strategy calculations
  - Parameter validation for all strategy-specific parameters
  - Pre-compiled regex patterns for improved performance
- **Comprehensive Edge Case Tests**: 77 new tests covering parameter validation, type errors, Unicode handling, and boundary conditions

### 0.3.0
- **Zero Dependencies**: Core library now works without any external dependencies
- **New Markov Chain Strategy**: Character-level Markov chain trained on 300K+ English words
- **New N-gram Frequency Strategy**: Trigram analysis using 2000 most common English trigrams
- **New Word Lookup Strategy**: 50K embedded English word dictionary (replaces pyspellchecker dependency)
- **Embedded Training Data**: Pre-computed bigrams, trigrams, and word sets included in package
- **Optional Dependencies**: FastText and pyspellchecker moved to optional extras
- **Lightweight Package**: ~190KB wheel size (well under 5MB limit)
- **Data Source**: Training data from Peter Norvig's word frequency list (MIT licensed)

### 0.2.0
- **New Keyboard Pattern Strategy**: Best-performing strategy with 69.9% F1 score
- **New Vowel Ratio Strategy**: Highest precision (95.45%) with consonant cluster detection
- **EnsembleDetector**: Built-in ensemble with majority/average/weighted voting
- **Enhanced Entropy Strategy**: Added bigram frequency analysis using common English bigrams
- **Enhanced Pattern Matching**: Added keyboard row patterns, consonant clusters, alternating patterns
- **Input Validation**: Validates threshold (0-1) and threads parameters
- **Type Hints**: Full type annotation throughout the codebase
- **Regression Tests**: 117 test cases across 20 categories with benchmarking
- **Performance**: Regex patterns compiled once at initialization

### 0.1.0
- Initial release with 7 detection strategies
- Scikit-learn-like interface
- Probability scoring
- Modular architecture
