"""Profanity word lists. Source of truth for profanity.json.

Seed: the LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise Bad Words"
(English), Shutterstock, licensed CC-BY-4.0
(https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words).
Filtered to profanity and slurs (sexual-health and anatomical vocabulary that
appears in ordinary text is excluded), normalised, and extended with common
compounds. Entries are lowercase ASCII letters only.
"""

from typing import Any, Dict, Tuple

ATTRIBUTION = (
    "Seeded from the LDNOOBW English list (Shutterstock), CC-BY-4.0, "
    "filtered and extended by the pygarble maintainers."
)

PROFANITY_STRONG: Tuple[str, ...] = tuple(
    sorted(
        {
            "arse",
            "arsehole",
            "arseholes",
            "ass",
            "asshat",
            "asshole",
            "assholes",
            "asswipe",
            "bastard",
            "bastards",
            "bitch",
            "bitches",
            "bitchy",
            "bollocks",
            "bullshit",
            "bullshitter",
            "chink",
            "chinks",
            "clit",
            "cock",
            "cocks",
            "cocksucker",
            "cocksuckers",
            "coon",
            "coons",
            "cum",
            "cunt",
            "cunts",
            "dago",
            "dickhead",
            "dickheads",
            "dicks",
            "dipshit",
            "douchebag",
            "douchebags",
            "dumbass",
            "dumbasses",
            "dyke",
            "dykes",
            "fag",
            "faggot",
            "faggots",
            "fags",
            "fuck",
            "fucked",
            "fucker",
            "fuckers",
            "fuckin",
            "fucking",
            "fucks",
            "fuckwit",
            "goddamn",
            "gook",
            "gooks",
            "horseshit",
            "jackass",
            "jackasses",
            "jizz",
            "kike",
            "kikes",
            "kunt",
            "motherfucker",
            "motherfuckers",
            "motherfucking",
            "nigga",
            "niggas",
            "nigger",
            "niggers",
            "paki",
            "pakis",
            "piss",
            "pissed",
            "pisses",
            "pissing",
            "prick",
            "pricks",
            "pussies",
            "pussy",
            "raghead",
            "retard",
            "retarded",
            "retards",
            "shit",
            "shite",
            "shitfaced",
            "shithead",
            "shitheads",
            "shits",
            "shitty",
            "slut",
            "sluts",
            "spic",
            "spics",
            "tits",
            "titties",
            "tranny",
            "trannies",
            "twat",
            "twats",
            "wank",
            "wanker",
            "wankers",
            "wetback",
            "wetbacks",
            "whore",
            "whores",
            "wog",
            "wogs",
            "wop",
            "wops",
        }
    )
)

PROFANITY_MILD: Tuple[str, ...] = tuple(
    sorted(
        {
            "bugger",
            "crap",
            "crappy",
            "damn",
            "damned",
            "dammit",
            "darn",
            "douche",
            "frigging",
            "jerkoff",
            "pissoff",
            "sod",
            "sodding",
            "turd",
            "wanky",
        }
    )
)

PHRASES: Tuple[Tuple[str, ...], ...] = (
    ("son", "of", "a", "bitch"),
    ("piece", "of", "shit"),
    ("mother", "fucker"),
    ("bull", "shit"),
    ("jack", "ass"),
    ("dumb", "ass"),
)

# Strong words also matched inside longer tokens (fuckwit, shitposting).
# Each must be at least four letters and occur inside no ENGLISH_WORDS entry.
# "cunt", "shit" and "wank" are left out because they occur inside clean
# English words: scunthorpe, matsushita and swank.
EMBEDDED: Tuple[str, ...] = ("fuck",)


def export() -> Dict[str, Any]:
    from .normalize import LEET_MAP

    return {
        "strong": list(PROFANITY_STRONG),
        "mild": list(PROFANITY_MILD),
        "phrases": [list(p) for p in PHRASES],
        "embedded": list(EMBEDDED),
        "leet_map": dict(LEET_MAP),
        "attribution": ATTRIBUTION,
    }
