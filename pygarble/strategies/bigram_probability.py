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
    # Not just rare - they simply don't occur in valid words
    IMPOSSIBLE_BIGRAMS = frozenset({
        # Q rules - 'q' is almost always followed by 'u'
        "qb", "qc", "qd", "qf", "qg", "qh", "qj", "qk", "ql", "qm",
        "qn", "qp", "qr", "qs", "qt", "qv", "qw", "qx", "qy", "qz",
        # Doubled consonants that never double
        "jj", "kk", "qq", "vv", "ww", "xx", "hh",
        # These combinations are essentially impossible
        "bx", "cx", "dx", "fx", "gx", "hx", "jx", "kx", "lx", "mx",
        "nx", "px", "qx", "rx", "sx", "tx", "vx", "wx", "zx",
        "xj", "xk", "xq", "xz",
        "bq", "cq", "dq", "fq", "gq", "hq", "jq", "kq", "lq", "mq",
        "nq", "pq", "rq", "sq", "tq", "vq", "wq", "xq", "yq", "zq",
        "fz", "gz", "hz", "jz", "kz", "pz", "qz", "vz", "wz", "xz",
        "bj", "cj", "dj", "fj", "gj", "hj", "kj", "lj", "mj", "nj",
        "pj", "qj", "rj", "sj", "tj", "vj", "wj", "xj", "yj", "zj",
        "jb", "jc", "jd", "jf", "jg", "jh", "jk", "jl", "jm", "jn",
        "jp", "jq", "jr", "js", "jt", "jv", "jw", "jx", "jy", "jz",
        "zb", "zc", "zd", "zf", "zg", "zj", "zk", "zm", "zn", "zp",
        "zq", "zr", "zs", "zt", "zv", "zw", "zx",
        # More impossible combinations
        "vk", "vq", "vx", "kv", "kg", "gk",
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
        # Extract only alphabetic characters, lowercase
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0  # Too short to analyze

        # Count impossible bigrams
        impossible_count = 0
        total_bigrams = 0

        for i in range(len(alpha_text) - 1):
            bigram = alpha_text[i:i+2]
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

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
