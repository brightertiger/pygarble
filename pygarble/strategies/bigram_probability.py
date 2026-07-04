"""
Bigram Probability Strategy for detecting garbled text.

Detects impossible or extremely rare character pairs that almost never
occur in English. Conservative thresholds to minimize false positives.
"""

from .base import BaseStrategy


class BigramProbabilityStrategy(BaseStrategy):
    """
    Detects garbled text by identifying impossible character bigrams.

    Uses a curated list of letter pairs that essentially never appear
    in English words. Very conservative to avoid flagging typos.
    """

    # These bigrams are virtually impossible in English
    # Not just rare - they simply don't occur in valid words.
    # Every entry is verified against pygarble.data.bigrams: only pairs
    # with a corpus log-probability <= -7 are eligible. Pairs that occur
    # in real words were removed: sq (square), nj (enjoy), nx (lynx),
    # fj (fjord), dj (adjust), hh (withhold), ww (www), gz (zigzag),
    # kg (units), vv (savvy), xx/jj/kk/qq and most j-/z-adjacent pairs
    # whose corpus probability shows they do occur.
    IMPOSSIBLE_BIGRAMS = frozenset({
        # Q rules - 'q' is almost always followed by 'u'
        "qg", "qh", "qj", "qk", "qn", "qx", "qy", "qz",
        # X preceded by a consonant that never precedes it
        "bx", "cx", "dx", "fx", "gx", "hx", "jx", "kx", "lx", "mx",
        "px", "rx", "sx", "tx", "vx", "wx", "zx",
        "xj", "xk", "xq", "xz",
        # Q preceded by consonants
        "bq", "cq", "dq", "fq", "gq", "hq", "jq", "kq", "lq", "mq",
        "nq", "pq", "rq", "tq", "vq", "wq", "yq", "zq",
        # Z pairs
        "fz", "hz", "jz", "kz", "pz", "vz", "wz", "zf",
        # J pairs
        "cj", "gj", "hj", "lj", "mj", "pj", "rj", "sj", "tj", "vj",
        "yj", "jg", "jy",
        # More impossible combinations
        "vk", "kv", "gk",
    })

    def __init__(
        self,
        threshold: float = 0.3,
        min_length: int = 4,
        **kwargs
    ):
        """
        Initialize the bigram probability strategy.

        Args:
            threshold: Ratio of impossible bigrams to flag (default 0.3 = 30%)
            min_length: Minimum text length to analyze (default 4)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_length = min_length

    def _predict_proba_impl(self, text: str) -> float:
        # Bigrams are only formed within words - never across word
        # boundaries ("of fjords" must not create the bigram "ff").
        words = [
            "".join(c.lower() for c in word if c.isalpha())
            for word in self._fold_diacritics(text).split()
        ]

        if sum(len(w) for w in words) < self.min_length:
            return 0.0  # Too short to analyze

        # Count impossible bigrams
        impossible_count = 0
        total_bigrams = 0

        for word in words:
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                total_bigrams += 1
                if bigram in self.IMPOSSIBLE_BIGRAMS:
                    impossible_count += 1

        if total_bigrams == 0:
            return 0.0

        ratio = impossible_count / total_bigrams

        # Scale to probability: if ratio >= threshold, return 1.0
        # Use a conservative scaling
        if ratio >= self.threshold:
            return 1.0
        elif ratio > 0:
            # Return scaled probability, but keep it low for occasional hits
            return min(0.6, ratio / self.threshold * 0.6)
        return 0.0
