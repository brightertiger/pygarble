"""
Rare Trigram Density Strategy for detecting garbled text.

Detects text with a high density of trigrams (3-letter sequences) that
essentially never occur in English. Uses a curated list of impossible
combinations for high precision.
"""

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
        "jjj", "kkk", "qqq", "vvv", "www", "xxx", "zzz",
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
        # Extract alphabetic characters only
        alpha_text = "".join(c.lower() for c in text if c.isalpha())

        if len(alpha_text) < self.min_length:
            return 0.0

        # Count impossible trigrams
        impossible_count = 0
        total_trigrams = len(alpha_text) - 2

        if total_trigrams <= 0:
            return 0.0

        for i in range(total_trigrams):
            trigram = alpha_text[i:i+3]
            if trigram in self.IMPOSSIBLE_TRIGRAMS:
                impossible_count += 1

        if impossible_count == 0:
            return 0.0

        ratio = impossible_count / total_trigrams

        # Scale to probability
        if ratio >= self.threshold:
            return min(1.0, 0.5 + ratio)

        # Low score for occasional matches
        return ratio / self.threshold * 0.4

    def _predict_impl(self, text: str) -> bool:
        return self._predict_proba_impl(text) >= 0.5
