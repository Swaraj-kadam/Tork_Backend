# api/safety.py

import re

# List of dangerous categories
HARMFUL_KEYWORDS = [
    # Weapons / harm
    r"make a bomb", r"build a bomb", r"explosive", r"weapon",
    r"molotov", r"napalm", r"gunpowder", r"dynamite",
    r"how to kill", r"how to harm", r"how to poison",

    # Cybercrime
    r"hacking", r"hack into", r"bypass login", r"sql injection",
    r"malware", r"ransomware", r"virus creation",r"hack"

    # Self-harm
    r"kill myself", r"end my life", r"self harm",

    # Illegal actions
    r"fake passport", r"counterfeit", r"buy drugs", r"drug recipe",
    r"steal.*credit card", r"carding", r"fraud",

    # Extremism / terrorism
    r"join isis", r"terrorist", r"extremist", r"mass shooting",
]

def is_harmful_query(query: str) -> bool:
    """Simple rule-based harmful content detector."""

    q = query.lower().strip()

    # Exact pattern matches
    for pattern in HARMFUL_KEYWORDS:
        if re.search(pattern, q):
            return True

    # Extra semantic checks
    if "how to make" in q and ("weapon" in q or "explosive" in q or "bomb" in q):
        return True

    if ("guide" in q or "instructions" in q) and ("kill" in q or "harm" in q):
        return True

    return False
