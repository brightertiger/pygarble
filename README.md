# pygarble

**Detect gibberish, garbled text, and corrupted content with high accuracy using advanced machine learning techniques.**

pygarble is a powerful Python library designed to identify nonsensical, garbled, or corrupted text content that often appears in data processing pipelines, user inputs, or automated systems. Whether you're dealing with random character sequences, encoding errors, keyboard mashing, or corrupted data streams, pygarble provides multiple detection strategies to filter out unwanted content and maintain data quality. The library uses statistical analysis, entropy calculations, pattern matching, n-gram analysis, and language detection to distinguish between meaningful text and gibberish with configurable sensitivity levels.

## Features

- **9 Detection Strategies**: Choose from multiple garble detection algorithms including keyboard pattern detection
- **Ensemble Detector**: Combine multiple strategies for higher accuracy with voting mechanisms
- **Scikit-learn Interface**: Familiar `predict()` and `predict_proba()` methods
- **Configurable Thresholds**: Adjust sensitivity for each strategy
- **Probability Scores**: Get confidence scores for garble detection
- **Input Validation**: Built-in validation for thresholds and parameters
- **Type Hints**: Full type annotation support throughout the codebase
- **Modular Design**: Easy to extend with new detection strategies
- **Enterprise Ready**: Support for offline model paths and restricted environments
- **Smart Edge Cases**: Automatically detects extremely long strings without whitespace (like base64 data)

## Installation

You can install pygarble using pip:

```bash
pip install pygarble
```

## Quick Start

```python
from pygarble import GarbleDetector, Strategy, EnsembleDetector

# Create a detector with the best-performing strategy
detector = GarbleDetector(Strategy.KEYBOARD_PATTERN, threshold=0.5)

# Detect garbled text
texts = ["Hello world", "asdfghjkl", "qwertyuiop"]
results = detector.predict(texts)
print(results)  # [False, True, True]

# Get probability scores
probabilities = detector.predict_proba(texts)
print(probabilities)  # [0.1, 0.85, 0.92]

# Use ensemble for best accuracy
ensemble = EnsembleDetector()
print(ensemble.predict("keyboard mashing test asdfgh"))  # True
```

## Benchmark Results

Based on 117 real-world test cases across 20 categories:

| Strategy | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| **keyboard_pattern** | 73.50% | 70.59% | 69.23% | **69.90%** |
| **vowel_ratio** | 72.65% | 95.45% | 40.38% | 56.76% |
| **ensemble** | 71.79% | 95.24% | 38.46% | 54.79% |
| english_word_validation | 61.54% | 60.61% | 38.46% | 47.06% |
| entropy_based | 64.96% | 72.00% | 34.62% | 46.75% |
| statistical_analysis | 63.25% | 69.57% | 30.77% | 42.67% |

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

### 5. English Word Validation (`ENGLISH_WORD_VALIDATION`)

**Implementation Logic**: Validates words against an English dictionary using pyspellchecker.

```python
detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION, threshold=0.5)

# Examples
detector.predict("hello world")              # False - valid words
detector.predict("asdfghjkl qwertyuiop")    # True - invalid words
```

### 6. Character Frequency (`CHARACTER_FREQUENCY`)

**Implementation Logic**: Analyzes character frequency distribution. Garbled text often has skewed distributions.

```python
detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY, threshold=0.5)

# Examples
detector.predict("aaaaaaa")         # True - high 'a' frequency
detector.predict("normal text")     # False - balanced distribution
```

### 7. Statistical Analysis (`STATISTICAL_ANALYSIS`)

**Implementation Logic**: Analyzes the ratio of alphabetic to non-alphabetic characters.

```python
detector = GarbleDetector(Strategy.STATISTICAL_ANALYSIS, threshold=0.5)

# Examples
detector.predict("123456789")       # True - no alphabetic chars
detector.predict("normal text")     # False - mostly alphabetic
```

### 8. Word Length (`WORD_LENGTH`)

**Implementation Logic**: Checks average word length against normal English patterns.

```python
detector = GarbleDetector(Strategy.WORD_LENGTH, threshold=0.5)

# Examples
detector.predict("supercalifragilistic")  # True - very long word
detector.predict("short words here")       # False - normal lengths
```

### 9. Language Detection (`LANGUAGE_DETECTION`)

**Implementation Logic**: Uses FastText to detect if text is in the expected language.

```python
detector = GarbleDetector(
    Strategy.LANGUAGE_DETECTION,
    target_language='en',
    threshold=0.5
)

# Examples
detector.predict("Hello world")            # False - English
detector.predict("asdfghjkl")              # True - not English
detector.predict("Bonjour le monde")       # True - French, not English
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

### Language Detection (Offline)

```python
detector = GarbleDetector(
    Strategy.LANGUAGE_DETECTION,
    model_path='/path/to/lid.176.bin',
    target_language='en'
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
    voting: str = "majority",  # "majority", "average", "weighted"
    weights: Optional[List[float]] = None,
    threads: Optional[int] = None,
    **kwargs
)
```

### Strategy Enum

```python
class Strategy(Enum):
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    LANGUAGE_DETECTION = "language_detection"
    ENGLISH_WORD_VALIDATION = "english_word_validation"
    VOWEL_RATIO = "vowel_ratio"
    KEYBOARD_PATTERN = "keyboard_pattern"
```

## Architecture

```
pygarble/
├── __init__.py
├── core.py                      # GarbleDetector & EnsembleDetector
└── strategies/
    ├── base.py                  # BaseStrategy with shared utilities
    ├── character_frequency.py
    ├── word_length.py
    ├── pattern_matching.py      # Regex patterns + keyboard detection
    ├── statistical_analysis.py
    ├── entropy_based.py         # Shannon entropy + bigram analysis
    ├── language_detection.py    # FastText integration
    ├── english_word_validation.py
    ├── vowel_ratio.py           # Vowel analysis + consonant clusters
    └── keyboard_pattern.py      # N-gram + keyboard row detection
```

## Dependencies

- Python 3.8+
- numpy>=1.21.0
- fasttext-wheel>=0.9.2 (for language detection)
- pyspellchecker>=0.8.0 (for English word validation)

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

### 0.2.0 (Current)
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
