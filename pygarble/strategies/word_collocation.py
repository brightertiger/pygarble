"""
Word Collocation Strategy for detecting garbled text.

Natural English text contains common adjacent word pairs (collocations)
like "of the", "in the", "it is". Garbled text almost never produces
these common pairings.
"""

import re
from typing import Any, List, Tuple

from .base import BaseStrategy


class WordCollocationStrategy(BaseStrategy):
    """
    Detect garbled text by checking for common English word bigrams.

    Uses a hardcoded set of common English word pairs. If multi-word
    text contains zero common collocations, that is suspicious.

    Parameters
    ----------
    min_words : int, optional
        Minimum word count for analysis. Default is 5.

    zero_collocation_min_words : int, optional
        Minimum words to trigger zero-collocation signal. Default is 8.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.WORD_COLLOCATION)
    >>> detector.predict("It is going to be a long day")
    False
    """

    COMMON_COLLOCATIONS = frozenset({
        # Preposition + article
        ("of", "the"), ("in", "the"), ("to", "the"), ("on", "the"),
        ("for", "the"), ("at", "the"), ("by", "the"), ("from", "the"),
        ("with", "the"), ("into", "the"), ("about", "the"),
        ("through", "the"), ("over", "the"), ("after", "the"),
        ("under", "the"), ("between", "the"), ("during", "the"),
        ("before", "the"), ("around", "the"), ("across", "the"),
        # Preposition/verb + "a"
        ("of", "a"), ("in", "a"), ("to", "a"), ("is", "a"),
        ("was", "a"), ("as", "a"), ("for", "a"), ("with", "a"),
        ("on", "a"), ("at", "a"),
        # "the" + common nouns/adjectives
        ("the", "first"), ("the", "same"), ("the", "other"),
        ("the", "most"), ("the", "new"), ("the", "world"),
        ("the", "best"), ("the", "next"), ("the", "last"),
        ("the", "end"), ("the", "time"), ("the", "way"),
        # "a" + common adjectives
        ("a", "few"), ("a", "new"), ("a", "lot"), ("a", "good"),
        ("a", "great"), ("a", "long"), ("a", "little"), ("a", "very"),
        ("a", "large"), ("a", "small"), ("a", "single"),
        # Pronoun + verb
        ("it", "is"), ("it", "was"), ("it", "has"), ("it", "will"),
        ("it", "would"), ("it", "can"),
        ("i", "have"), ("i", "am"), ("i", "was"), ("i", "think"),
        ("i", "will"), ("i", "would"), ("i", "can"), ("i", "do"),
        ("he", "was"), ("he", "had"), ("he", "is"), ("he", "said"),
        ("she", "was"), ("she", "had"), ("she", "is"), ("she", "said"),
        ("we", "have"), ("we", "are"), ("we", "can"), ("we", "will"),
        ("they", "are"), ("they", "have"), ("they", "were"),
        ("they", "will"), ("they", "had"),
        ("you", "can"), ("you", "are"), ("you", "have"),
        ("you", "will"), ("you", "want"),
        ("who", "is"), ("who", "was"), ("who", "are"),
        ("what", "is"), ("what", "are"),
        ("there", "is"), ("there", "are"), ("there", "was"),
        ("there", "were"),
        # Verb patterns
        ("to", "be"), ("to", "do"), ("to", "have"), ("to", "make"),
        ("to", "get"), ("to", "go"), ("to", "take"), ("to", "see"),
        ("to", "know"), ("to", "find"), ("to", "give"),
        ("has", "been"), ("have", "been"), ("had", "been"),
        ("will", "be"), ("would", "be"), ("can", "be"),
        ("could", "be"), ("should", "be"), ("may", "be"),
        ("would", "have"), ("could", "have"), ("should", "have"),
        ("must", "have"), ("will", "have"),
        # Negation
        ("do", "not"), ("does", "not"), ("did", "not"),
        ("is", "not"), ("was", "not"), ("are", "not"),
        ("have", "not"), ("has", "not"), ("had", "not"),
        ("will", "not"), ("would", "not"), ("can", "not"),
        ("could", "not"), ("should", "not"),
        # Comparison/quantifier
        ("as", "well"), ("as", "much"), ("as", "many"),
        ("such", "as"), ("such", "a"),
        ("more", "than"), ("less", "than"), ("rather", "than"),
        ("other", "than"),
        ("one", "of"), ("some", "of"), ("all", "of"),
        ("most", "of"), ("each", "of"), ("many", "of"),
        ("part", "of"), ("out", "of"), ("kind", "of"),
        ("because", "of"), ("instead", "of"),
        # Common phrases
        ("in", "order"), ("as", "if"), ("even", "if"),
        ("so", "that"), ("in", "which"), ("at", "least"),
        ("at", "all"), ("no", "longer"), ("as", "long"),
        ("up", "to"), ("due", "to"), ("according", "to"),
        ("able", "to"), ("used", "to"), ("going", "to"),
        ("have", "to"), ("has", "to"), ("had", "to"),
        ("need", "to"), ("want", "to"), ("try", "to"),
        # Conjunction patterns
        ("is", "the"), ("was", "the"), ("are", "the"),
        ("and", "the"), ("and", "a"), ("and", "it"),
        ("and", "he"), ("and", "she"), ("and", "they"),
        ("and", "then"), ("and", "that"), ("and", "this"),
        ("but", "the"), ("but", "it"), ("but", "he"),
        ("or", "the"), ("or", "a"),
        # Common verb + determiner patterns (very frequent in English)
        ("see", "the"), ("see", "a"), ("get", "the"), ("get", "a"),
        ("make", "the"), ("make", "a"), ("take", "the"), ("take", "a"),
        ("give", "the"), ("give", "a"), ("find", "the"), ("find", "a"),
        ("read", "the"), ("read", "a"), ("send", "the"), ("send", "a"),
        ("use", "the"), ("use", "a"), ("know", "the"), ("know", "a"),
        ("like", "the"), ("like", "a"), ("need", "the"), ("need", "a"),
        ("keep", "the"), ("keep", "a"), ("leave", "the"), ("leave", "a"),
        ("put", "the"), ("put", "a"), ("set", "the"), ("set", "a"),
        ("tell", "the"), ("tell", "a"), ("ask", "the"), ("ask", "a"),
        ("try", "the"), ("try", "a"), ("call", "the"), ("call", "a"),
        ("run", "the"), ("run", "a"), ("let", "the"), ("let", "a"),
        ("check", "the"), ("check", "a"),
        ("review", "the"), ("review", "a"),
        ("open", "the"), ("open", "a"),
        ("close", "the"), ("close", "a"),
        ("start", "the"), ("start", "a"),
        ("follow", "the"), ("follow", "a"),
        ("provide", "the"), ("provide", "a"),
        ("include", "the"), ("include", "a"),
        ("consider", "the"), ("consider", "a"),
        ("bring", "the"), ("bring", "a"),
        ("change", "the"), ("change", "a"),
        # Past tense verb + determiner
        ("saw", "the"), ("saw", "a"), ("got", "the"), ("got", "a"),
        ("made", "the"), ("made", "a"), ("took", "the"), ("took", "a"),
        ("gave", "the"), ("gave", "a"), ("found", "the"), ("found", "a"),
        ("said", "the"), ("said", "a"), ("told", "the"), ("told", "a"),
        ("left", "the"), ("left", "a"), ("called", "the"),
        ("used", "the"), ("used", "a"),
        ("asked", "the"), ("asked", "a"),
        # Verb + pronoun patterns
        ("tell", "me"), ("give", "me"), ("send", "me"), ("show", "me"),
        ("let", "me"), ("ask", "me"), ("told", "me"), ("gave", "me"),
        ("tell", "him"), ("give", "him"), ("tell", "her"), ("give", "her"),
        ("tell", "them"), ("give", "them"), ("tell", "us"), ("give", "us"),
        ("provide", "your"), ("send", "your"), ("check", "your"),
        # "not" + common patterns
        ("not", "the"), ("not", "a"), ("not", "be"), ("not", "have"),
        ("not", "only"), ("not", "just"), ("not", "sure"),
        # Adverb patterns
        ("also", "the"), ("also", "a"), ("just", "the"), ("just", "a"),
        ("only", "the"), ("only", "a"), ("still", "the"), ("still", "a"),
        # "this/that" patterns
        ("this", "is"), ("this", "was"), ("that", "the"),
        ("that", "is"), ("that", "was"), ("that", "it"),
        ("that", "he"), ("that", "she"), ("that", "they"),
        ("that", "we"), ("that", "you"),
    })

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_words = kwargs.get("min_words", 8)
        self.zero_collocation_min_words = kwargs.get(
            "zero_collocation_min_words", 12
        )

        if self.min_words < 2:
            raise ValueError("min_words must be at least 2")

    def _tokenize(self, text: str) -> List[str]:
        """Extract lowercase alphabetic words."""
        return re.findall(r"[a-zA-Z]+", text.lower())

    def _get_bigrams(
        self, words: List[str]
    ) -> List[Tuple[str, str]]:
        """Generate adjacent word pairs."""
        return [
            (words[i], words[i + 1]) for i in range(len(words) - 1)
        ]

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize(text)

        if len(words) < self.min_words:
            return 0.0

        # Require enough substantial words (>= 3 chars) to avoid
        # false positives from hex strings, OCR fragments, etc.
        substantial = sum(1 for w in words if len(w) >= 3)
        if substantial < self.min_words // 2:
            return 0.0

        bigrams = self._get_bigrams(words)
        if not bigrams:
            return 0.0

        hit_count = sum(
            1 for bg in bigrams if bg in self.COMMON_COLLOCATIONS
        )
        total_bigrams = len(bigrams)
        hit_ratio = hit_count / total_bigrams

        # Zero collocations: scale with text length
        if hit_count == 0:
            if len(words) >= 20:
                return 0.85
            if len(words) >= self.zero_collocation_min_words:
                return 0.7
            # Below threshold: too short to be confident
            return 0.3

        # Very low collocation rate in long text
        if len(words) >= 15 and hit_ratio < 0.02:
            return 0.55

        return 0.0

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
