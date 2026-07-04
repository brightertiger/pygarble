"""
Letter Position Strategy for detecting garbled text.

Detects letters appearing in positions where they never occur in English.
Very conservative - only flags truly impossible positions.
"""

from .base import BaseStrategy


class LetterPositionStrategy(BaseStrategy):
    """
    Detects garbled text by identifying letters in impossible positions.

    Certain letters have positional constraints in English:
    - 'q' is almost always followed by 'u'
    - Some letters never end words
    - Some combinations never start words
    """

    # Letters that essentially NEVER end English words
    # Being very conservative - only including truly impossible endings
    # Note: "q" excluded - proper nouns like Iraq, Compaq end in 'q'
    NEVER_END = frozenset({
        "j",  # No standard English words end in 'j'
    })

    # Letter pairs that never START English words
    # NOTE: Being VERY conservative - only truly impossible starts
    # Valid but unusual: gn (gnocchi), pn (pneumonia), pt (pterodactyl),
    #   ps (psychology), ts (tsar), ht (technical), etc.
    NEVER_START = frozenset({
        "bw", "bx",
        "cb", "cd", "cf", "cg", "cj", "ck", "cm", "cn", "cp", "cq",
        "cv", "cw", "cx", "cz",
        "db", "dc", "df", "dg", "dk", "dl", "dm", "dn", "dp",
        "dq", "dt", "dx", "dz",
        # Skip "dj" - valid in djinn, Djibouti; "dv" - Dvorak
        "fc", "fd", "fg", "fj", "fk", "fm", "fn", "fp", "fq",
        "fv", "fw", "fx", "fz",
        # Skip "fb" - FBI and similar acronyms
        "gb", "gc", "gd", "gf", "gj", "gk", "gm", "gp", "gq",
        "gv", "gw", "gx", "gz",
        # Skip "gn" - valid in gnocchi, gnat, gnaw, gnome
        "hb", "hc", "hd", "hf", "hg", "hj", "hk", "hl", "hn",
        "hp", "hq", "hv", "hw", "hx", "hz",
        # Skip "hs", "ht" - valid in technical contexts
        # Skip "hm" - hmm; "hr" - hryvnia, HR acronym
        "jb", "jc", "jd", "jf", "jg", "jh", "jj", "jk", "jl", "jm",
        "jn", "jp", "jq", "jr", "js", "jt", "jv", "jw", "jx", "jy", "jz",
        "kb", "kc", "kd", "kf", "kg", "kj", "kk", "km", "kp", "kq",
        "kt", "kv", "kx", "kz",
        # Skip "ks" - valid in some words
        "lc", "ld", "lf", "lg", "lh", "lj", "lk", "lm", "ln",
        "lp", "lq", "lr", "ls", "lt", "lv", "lw", "lx", "lz",
        # Skip "lb" - lbs
        "md", "mf", "mg", "mh", "mj", "mk", "ml", "mm",
        "mp", "mq", "mt", "mv", "mw", "mx", "mz",
        # Skip "ms" - valid in technical
        # Skip "mb" - mbira; "mc" - McDonald; "mr" - Mr, Mrs
        "nc", "nd", "nf", "nh", "nj", "nl", "nm",
        "nn", "np", "nq", "nr", "ns", "nv", "nw", "nx", "nz",
        # Skip "ng" - Nguyen; "nb" - NBA/NBC; "nk" - NKVD; "nt" - nth
        "pb", "pc", "pd", "pf", "pg", "pj", "pk", "pm", "pp", "pq",
        "pv", "pw", "px", "pz",
        # Skip "pn" - valid in pneumonia, pneumatic
        # Skip "ps" - valid in psychology, psalm
        # Skip "pt" - valid in pterodactyl
        "qa", "qb", "qc", "qd", "qe", "qf", "qg", "qh", "qj", "qk",
        "ql", "qm", "qn", "qo", "qp", "qq", "qr", "qs", "qt", "qv",
        "qw", "qx", "qy", "qz",
        "rb", "rc", "rd", "rf", "rg", "rj", "rk", "rl", "rm", "rn",
        "rp", "rq", "rr", "rs", "rt", "rv", "rw", "rx", "rz",
        "sb", "sd", "sf", "sg", "sj", "sv", "sx", "sz",
        "tb", "td", "tf", "tg", "tj", "tk", "tl", "tm", "tn",
        "tp", "tq", "tt", "tx", "tz",
        # Skip "tc", "ts" - valid in technical, tsar; "tv" - TV
        "vc", "vd", "vf", "vg", "vh", "vj", "vk", "vl", "vm", "vn",
        "vp", "vq", "vr", "vs", "vt", "vv", "vw", "vx", "vz",
        "wb", "wc", "wd", "wf", "wg", "wj", "wk", "wl", "wm", "wn",
        "wp", "wq", "ws", "wt", "wv", "ww", "wx", "wz",
        "xc", "xd", "xf", "xg", "xh", "xj", "xk", "xl", "xm",
        "xn", "xp", "xq", "xr", "xs", "xt", "xv", "xw", "xx", "xz",
        # Skip "xb" - Xbox
        "yb", "yc", "yd", "yf", "yg", "yh", "yj", "yk", "yl", "ym",
        "yn", "yp", "yq", "yr", "ys", "yt", "yv", "yw", "yx", "yz",
        "zb", "zc", "zd", "zf", "zg", "zj", "zk", "zl", "zm", "zn",
        "zp", "zq", "zr", "zs", "zt", "zv", "zw", "zx", "zy", "zz",
    })

    def __init__(
        self,
        threshold: float = 0.25,
        min_word_length: int = 3,
        **kwargs
    ):
        """
        Initialize the letter position strategy.

        Args:
            threshold: Ratio of invalid positions to flag (default 0.25)
            min_word_length: Minimum word length to check (default 3)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.min_word_length = min_word_length

    def _extract_words(self, text: str):
        """Extract alphabetic words from text."""
        words = []
        current_word = []
        for c in text:
            if c.isalpha():
                current_word.append(c.lower())
            else:
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
        if current_word:
            words.append("".join(current_word))
        return [w for w in words if len(w) >= self.min_word_length]

    def _predict_proba_impl(self, text: str) -> float:
        words = self._extract_words(text)

        if not words:
            return 0.0

        violation_count = 0
        total_checks = 0

        for word in words:
            if len(word) < 2:
                continue

            # Check word ending
            total_checks += 1
            if word[-1] in self.NEVER_END:
                violation_count += 1

            # Check word start (first 2 letters)
            total_checks += 1
            if word[:2] in self.NEVER_START:
                violation_count += 1

        if total_checks == 0:
            return 0.0

        ratio = violation_count / total_checks

        # A single violation in a single word is weak evidence: proper
        # nouns and acronyms can break positional rules once. Require
        # at least 2 violations or 2 words before scoring above 0.5.
        if violation_count < 2 and len(words) < 2:
            if ratio > 0:
                return min(0.4, ratio / self.threshold * 0.4)
            return 0.0

        # Scale to probability
        if ratio >= self.threshold:
            return min(1.0, 0.5 + ratio)
        elif ratio > 0:
            return ratio / self.threshold * 0.4

        return 0.0
