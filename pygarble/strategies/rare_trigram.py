"""
Rare Trigram Density Strategy for detecting garbled text.

Detects text with a high density of trigrams (3-letter sequences) that
essentially never occur in English. Uses a curated list of impossible
combinations for high precision.
"""

import re

from .base import BaseStrategy


class RareTrigramStrategy(BaseStrategy):
    """
    Detects garbled text by identifying impossible trigram sequences.

    Uses a curated list of 3-letter combinations that essentially never
    appear in English words. Very conservative to avoid false positives.
    """

    # Trigrams that are virtually impossible in English
    # These patterns simply don't occur in valid English words
    IMPOSSIBLE_TRIGRAMS = frozenset({
        # Q without u patterns
        "qqa", "qqb", "qqc", "qqd", "qqe", "qqf", "qqg", "qqh", "qqi",
        "qqj", "qqk", "qql", "qqm", "qqn", "qqo", "qqp", "qqr", "qqs",
        "qqt", "qqu", "qqv", "qqw", "qqx", "qqy", "qqz",
        "qbq", "qcq", "qdq", "qfq", "qgq", "qhq", "qjq", "qkq", "qlq",
        "qmq", "qnq", "qpq", "qrq", "qsq", "qtq", "qvq", "qwq", "qxq",
        "qyq", "qzq",

        # Impossible consonant clusters
        "bxb", "bxc", "bxd", "bxf", "bxg", "bxh", "bxj", "bxk", "bxl",
        "bxm", "bxn", "bxp", "bxq", "bxr", "bxs", "bxt", "bxv", "bxw",
        "bxz",
        # NOTE: "www", "xxx", "zzz", "kkk" are deliberately absent --
        # they occur in real-world text (URLs, ratings, snoring, the
        # acronym) and caused false positives.
        "jjj", "qqq", "vvv",
        "jjk", "jjl", "jjm", "jjn", "jjp", "jjq", "jjr", "jjs", "jjt",
        "jjv", "jjw", "jjx", "jjy", "jjz",
        "xjx", "xkx", "xqx", "xvx", "xwx", "xzx",
        "zjz", "zkz", "zqz", "zvz", "zwz", "zxz",

        # Double rare letter + any
        "jxj", "jqj", "jzj", "qjq", "qxq", "qzq", "xjx", "xqx", "xzx",
        "zjx", "zqx", "zxj",

        # Impossible starting clusters
        "bwb", "bwc", "bwd", "bwf", "bwg", "bwh", "bwj", "bwk", "bwl",
        "bwm", "bwn", "bwp", "bwq", "bwr", "bws", "bwt", "bwv", "bww",
        "bwx", "bwz",
        "cxc", "cxd", "cxf", "cxg", "cxh", "cxj", "cxk", "cxl", "cxm",
        "cxn", "cxp", "cxq", "cxr", "cxs", "cxt", "cxv", "cxw", "cxz",
        "dxd", "dxf", "dxg", "dxh", "dxj", "dxk", "dxl", "dxm", "dxn",
        "dxp", "dxq", "dxr", "dxs", "dxt", "dxv", "dxw", "dxz",
        "fxf", "fxg", "fxh", "fxj", "fxk", "fxl", "fxm", "fxn", "fxp",
        "fxq", "fxr", "fxs", "fxt", "fxv", "fxw", "fxz",

        # Consecutive rare letters
        "jqx", "jqz", "jxq", "jxz", "jzq", "jzx",
        "qjx", "qjz", "qxj", "qxz", "qzj", "qzx",
        "xjq", "xjz", "xqj", "xqz", "xzj", "xzq",
        "zjq", "zjx", "zqj", "zqx", "zxj", "zxq",
    })

    def __init__(
        self,
        threshold: float = 0.15,
        min_length: int = 6,
        **kwargs
    ):
        """
        Initialize the rare trigram strategy.

        Args:
            threshold: Ratio of impossible trigrams to flag (default 0.15)
            min_length: Minimum text length to analyze (default 6)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_length = min_length

    def _predict_proba_impl(self, text: str) -> float:
        # Tokenize into words: trigrams must not cross word boundaries,
        # otherwise adjacent innocent words form phantom "impossible"
        # trigrams (e.g. "visit www" -> "tww").
        words = re.findall(r"[a-z]+", self._fold_diacritics(text).lower())

        alpha_length = sum(len(w) for w in words)
        if alpha_length < self.min_length:
            return 0.0

        # Count impossible trigrams within each word
        impossible_count = 0
        total_trigrams = 0

        for word in words:
            for i in range(len(word) - 2):
                total_trigrams += 1
                if word[i:i + 3] in self.IMPOSSIBLE_TRIGRAMS:
                    impossible_count += 1

        if total_trigrams <= 0 or impossible_count == 0:
            return 0.0

        ratio = impossible_count / total_trigrams

        # Scale to probability
        if ratio >= self.threshold:
            return min(1.0, 0.5 + ratio)

        # Low score for occasional matches
        return ratio / self.threshold * 0.4
