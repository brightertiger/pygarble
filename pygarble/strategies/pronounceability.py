"""
Pronounceability strategy for garble detection.

Analyzes if character sequences follow English phonotactic rules
(valid consonant clusters, syllable patterns, etc.).
"""

from typing import Any, Set

from .base import BaseStrategy


# Valid English consonant clusters at word START
# Based on English phonotactics
VALID_ONSET_CLUSTERS: Set[str] = {
    # Single consonants (all are valid)
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
    "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z",
    # Two-consonant clusters
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr",
    "gl", "gr", "pl", "pr", "sc", "sh", "sk", "sl", "sm",
    "sn", "sp", "st", "sw", "th", "tr", "tw", "wh", "wr",
    "kn", "gn", "pn", "ps", "ph", "qu",
    # Three-consonant clusters
    "chr", "phr", "sch", "scr", "shr", "spl", "spr", "squ",
    "str", "thr", "thw",
}

# Valid English consonant clusters at word END
VALID_CODA_CLUSTERS: Set[str] = {
    # Single consonants
    "b", "c", "d", "f", "g", "k", "l", "m", "n", "p",
    "r", "s", "t", "v", "x", "z",
    # Two-consonant clusters
    "ch", "ck", "ct", "ft", "lb", "lc", "ld", "lf", "lk",
    "ll", "lm", "ln", "lp", "ls", "lt", "lv", "lz",
    "mb", "mp", "ms", "nd", "ng", "nk", "ns", "nt", "nz",
    "pt", "rb", "rc", "rd", "rf", "rg", "rk", "rl", "rm",
    "rn", "rp", "rs", "rt", "rv", "rz", "sh", "sk", "sm",
    "sp", "ss", "st", "th", "ts", "xt", "zz",
    # Three-consonant clusters
    "lls", "lts", "nds", "ngs", "nks", "nts", "rbs", "rds",
    "rks", "rms", "rns", "rps", "rts", "sks", "sts", "tch",
    "dth", "fth", "nth", "pth", "xth",
}

# Completely invalid consonant combinations (never occur in English)
FORBIDDEN_CLUSTERS: Set[str] = {
    # Unpronounceable combinations
    "bk", "bv", "bz", "cb", "cd", "cf", "cg", "cj", "cm",
    "cn", "cp", "cv", "cz", "db", "dc", "df", "dg", "dj",
    "dk", "dl", "dm", "dn", "dp", "dt", "dv", "dx", "dz",
    "fb", "fc", "fd", "fg", "fj", "fk", "fm", "fn", "fp",
    "fv", "fw", "fx", "fz", "gb", "gc", "gd", "gf", "gj",
    "gk", "gm", "gp", "gt", "gv", "gw", "gx", "gz",
    "hb", "hc", "hd", "hf", "hg", "hh", "hj", "hk", "hl",
    "hm", "hn", "hp", "hs", "ht", "hv", "hw", "hx", "hz",
    # Note: "hr" removed - appears in valid clusters like "thr", "chr", "phr"
    "jb", "jc", "jd", "jf", "jg", "jh", "jj", "jk", "jl",
    "jm", "jn", "jp", "jr", "js", "jt", "jv", "jw", "jx", "jz",
    "kb", "kc", "kd", "kf", "kg", "kh", "kj", "kk", "kl",
    "km", "kp", "kq", "kr", "ks", "kt", "kv", "kw", "kx", "kz",
    "lj", "lq", "lx",
    "mf", "mg", "mj", "mk", "ml", "mq", "mr", "mt", "mv", "mw", "mx", "mz",
    "nf", "nj", "nl", "nm", "np", "nq", "nr", "nv", "nw", "nx",
    "pb", "pc", "pd", "pf", "pg", "pj", "pk", "pm", "pn", "pp",
    "pq", "pt", "pv", "pw", "px", "pz",
    "qb", "qc", "qd", "qe", "qf", "qg", "qh", "qi", "qj", "qk",
    "ql", "qm", "qn", "qo", "qp", "qq", "qr", "qs", "qt", "qv",
    "qw", "qx", "qy", "qz",
    "rj", "rq", "rx",
    "sb", "sd", "sf", "sg", "sj", "sr", "sv", "sx", "sz",
    "tb", "tc", "td", "tf", "tg", "tj", "tk", "tl", "tm",
    "tn", "tp", "tq", "tt", "tv", "tx", "tz",
    "vb", "vc", "vd", "vf", "vg", "vh", "vj", "vk", "vl",
    "vm", "vn", "vp", "vq", "vr", "vs", "vt", "vv", "vw", "vx", "vz",
    "wb", "wc", "wd", "wf", "wg", "wj", "wk", "wl", "wm",
    "wn", "wp", "wq", "ws", "wt", "wv", "ww", "wx", "wz",
    "xb", "xc", "xd", "xf", "xg", "xh", "xj", "xk", "xl",
    "xm", "xn", "xp", "xq", "xr", "xs", "xv", "xw", "xx", "xz",
    "yb", "yc", "yd", "yf", "yg", "yh", "yj", "yk", "yl",
    "ym", "yn", "yp", "yq", "yr", "ys", "yt", "yv", "yw", "yx", "yz",
    "zb", "zc", "zd", "zf", "zg", "zh", "zj", "zk", "zl",
    "zm", "zn", "zp", "zq", "zr", "zs", "zt", "zv", "zw", "zx",
}

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


class PronouncabilityStrategy(BaseStrategy):
    """
    Detect garbled text based on pronounceability.

    Analyzes if text follows English phonotactic rules:
    - Valid consonant clusters at word boundaries
    - Reasonable vowel distribution
    - No impossible consonant sequences

    Parameters
    ----------
    forbidden_cluster_threshold : int, optional
        Number of forbidden clusters to flag as garbled.
        Default is 2.

    min_word_length : int, optional
        Minimum word length to analyze for clusters.
        Default is 3.

    vowel_min_ratio : float, optional
        Minimum vowel ratio for pronounceable text.
        Default is 0.1.

    Examples
    --------
    >>> from pygarble import GarbleDetector, Strategy
    >>> detector = GarbleDetector(Strategy.PRONOUNCEABILITY)
    >>> detector.predict("xkcd qwfp zxcv")  # Unpronounceable
    True
    >>> detector.predict("hello world")
    False
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.forbidden_cluster_threshold = kwargs.get("forbidden_cluster_threshold", 2)
        self.min_word_length = kwargs.get("min_word_length", 3)
        self.vowel_min_ratio = kwargs.get("vowel_min_ratio", 0.1)

        if self.forbidden_cluster_threshold < 1:
            raise ValueError("forbidden_cluster_threshold must be at least 1")
        if self.min_word_length < 2:
            raise ValueError("min_word_length must be at least 2")
        if not 0.0 <= self.vowel_min_ratio <= 1.0:
            raise ValueError("vowel_min_ratio must be between 0.0 and 1.0")

    def _extract_consonant_clusters(self, word: str) -> list:
        """Extract all consonant clusters from a word."""
        clusters = []
        current_cluster = ""

        for char in word.lower():
            if char in CONSONANTS:
                current_cluster += char
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = ""

        # Don't forget trailing cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        return clusters

    def _get_word_onset(self, word: str) -> str:
        """Get consonant cluster at start of word."""
        onset = ""
        for char in word.lower():
            if char in CONSONANTS:
                onset += char
            elif char in VOWELS:
                break
            else:
                break
        return onset

    def _get_word_coda(self, word: str) -> str:
        """Get consonant cluster at end of word."""
        coda = ""
        for char in reversed(word.lower()):
            if char in CONSONANTS:
                coda = char + coda
            elif char in VOWELS:
                break
            else:
                break
        return coda

    def _count_forbidden_clusters(self, text: str) -> int:
        """Count forbidden consonant clusters in text."""
        count = 0
        words = text.lower().split()

        for word in words:
            # Only check alpha words
            alpha_word = "".join(c for c in word if c.isalpha())
            if len(alpha_word) < self.min_word_length:
                continue

            # Check all consonant clusters in the word
            clusters = self._extract_consonant_clusters(alpha_word)
            for cluster in clusters:
                # Check all bigrams within the cluster
                for i in range(len(cluster) - 1):
                    bigram = cluster[i:i+2]
                    if bigram in FORBIDDEN_CLUSTERS:
                        count += 1

        return count

    def _check_vowel_ratio(self, text: str) -> float:
        """Check if text has reasonable vowel distribution."""
        alpha_chars = [c.lower() for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0

        vowel_count = sum(1 for c in alpha_chars if c in VOWELS)
        ratio = vowel_count / len(alpha_chars)

        if ratio < self.vowel_min_ratio:
            # Very low vowel ratio = unpronounceable
            return 1.0 - (ratio / self.vowel_min_ratio)
        return 0.0

    def _check_onset_validity(self, text: str) -> float:
        """Check if word onsets are valid English clusters."""
        words = text.lower().split()
        invalid_count = 0
        checked_count = 0

        for word in words:
            alpha_word = "".join(c for c in word if c.isalpha())
            if len(alpha_word) < self.min_word_length:
                continue

            onset = self._get_word_onset(alpha_word)
            if len(onset) >= 2:
                checked_count += 1
                # Check if onset is valid or any prefix of it
                is_valid = any(
                    onset[:i+1] in VALID_ONSET_CLUSTERS
                    for i in range(len(onset))
                )
                if not is_valid and onset not in VALID_ONSET_CLUSTERS:
                    invalid_count += 1

        if checked_count == 0:
            return 0.0

        return invalid_count / checked_count

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        """
        Compute garble probability based on pronounceability.
        """
        if not text or len(text) < 3:
            return 0.0

        # Only analyze if text has enough alphabetic content
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) < 5:
            return 0.0

        scores = []

        # Check forbidden clusters
        forbidden_count = self._count_forbidden_clusters(text)
        if forbidden_count >= self.forbidden_cluster_threshold:
            forbidden_score = min(1.0, 0.6 + (forbidden_count * 0.1))
            scores.append(forbidden_score)

        # Check vowel ratio
        vowel_score = self._check_vowel_ratio(text)
        if vowel_score > 0:
            scores.append(vowel_score)

        # Check onset validity
        onset_score = self._check_onset_validity(text)
        if onset_score > 0.3:  # More than 30% invalid onsets
            scores.append(onset_score)

        # Combine scores
        if not scores:
            return 0.0

        # Use weighted combination favoring the strongest signal
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Blend max and average (max has more weight)
        return 0.7 * max_score + 0.3 * avg_score
