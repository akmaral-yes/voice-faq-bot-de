"""Input guards: regex PII masking and deterministic prompt-injection detection.

Both are intentionally simple demo implementations, not production-grade.
"""

import re

# --- PII masking -------------------------------------------------------------
#
# Customer/contract numbers: the real Deutsche Telekom identifier formats are
# NOT defined by this project. As a deliberately simplified demo rule, a number
# is only masked when it directly follows an explicit label such as
# "Kundennummer" or "Vertrags-Nr." (optionally with ":", "#", "ist", "lautet").
# Arbitrary long numbers without such a label are not masked.
#
# Order matters: IBAN and labelled numbers are masked before phone numbers,
# so their digits are not mistaken for a phone number.

_LABELLED_NUMBER = r"(?:\s*[:#]\s*|\s+(?:ist|lautet)\s+|\s+)(\d[\d\-/]{3,18}\d)\b"

PII_PATTERNS = [
    # Country code + 2 check digits + 11-30 alphanumerics, optionally in groups of 4.
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b", re.I)),
    ("KUNDENNUMMER", re.compile(r"\b(Kunden(?:nummer|-?Nr\.?))" + _LABELLED_NUMBER, re.I)),
    ("VERTRAGSNUMMER", re.compile(r"\b(Vertrags(?:nummer|-?Nr\.?))" + _LABELLED_NUMBER, re.I)),
    # German/Swiss: +49 / +41 / 0049 / 0041 (optionally "(0)") or a leading 0,
    # then an area/mobile prefix and at least one more digit group.
    ("TELEFONNUMMER", re.compile(
        r"(?<!\w)(?:(?:\+|00)(?:49|41)[\s\-/]?(?:\(0\)[\s\-/]?)?|0)\d{2,5}(?:[\s\-/]?\d{2,}){1,4}(?!\w)"
    )),
]


def mask_pii(text):
    """Replace PII with placeholders. Returns (masked_text, sorted list of PII types found)."""
    found = set()
    for pii_type, pattern in PII_PATTERNS:
        if pii_type in ("KUNDENNUMMER", "VERTRAGSNUMMER"):
            # keep the label, replace only the number
            replacement = lambda m, t=pii_type: m.group(0).replace(m.group(2), f"[{t}]")
        else:
            replacement = f"[{pii_type}]"
        text, count = pattern.subn(replacement, text)
        if count:
            found.add(pii_type)
    return text, sorted(found)


# --- Prompt-injection detection ----------------------------------------------
#
# Deterministic pattern matching on phrase combinations (not isolated words like
# "system" or "prompt", to limit false positives).
# KNOWN LIMITATION: this is easily bypassed by semantic rephrasing, typos,
# other languages or encodings. It only catches obvious attempts.

_GAP = r"(?:\s+\S+){0,4}\s+"  # up to 4 arbitrary words between key phrases

INJECTION_PATTERNS = [
    ("ignore_instructions", re.compile(
        r"\b(?:ignorier\w*|vergiss|missachte\w*)" + _GAP
        + r"(?:anweisungen|regeln|instruktionen|vorgaben)\b"
    )),
    ("ignore_instructions", re.compile(
        r"\b(?:ignore|disregard|forget)" + _GAP + r"(?:instructions|rules|prompts?)\b"
    )),
    ("reveal_system_prompt", re.compile(
        r"\b(?:zeig\w*|nenn\w*|gib|verrat\w*|wiederhol\w*|ausgeben)" + _GAP
        + r"(?:system[\s\-]?prompt|systemanweisung\w*|(?:deine|ihre) (?:anweisungen|instruktionen|regeln))\b"
    )),
    ("reveal_system_prompt", re.compile(
        r"\b(?:reveal|show|print|tell|repeat|output)" + _GAP
        + r"(?:system[\s\-]?prompt|(?:your|the) (?:instructions|initial prompt|rules))\b"
    )),
    # "du bist jetzt" + a role word, so "du bist jetzt schon der dritte Bot" is not blocked
    ("role_override", re.compile(
        r"\bdu bist (?:jetzt|ab sofort|ab jetzt|nun) (?:ein|eine|einer|kein|keine|mein|meine|nicht mehr|dan)\b"
    )),
    ("role_override", re.compile(r"\b(?:tu so,? als (?:ob|wärst) du|spiel\w* die rolle)\b")),
    ("role_override", re.compile(r"\b(?:you are now|from now on,? you|pretend (?:to be|you are)|act as)\b")),
    ("role_override", re.compile(r"\b(?:developer mode|entwicklermodus|jailbreak)\b")),
]


def detect_injection(text):
    """Return all matched reasons (e.g. ['injection:role_override']); empty list if none."""
    normalized = " ".join(text.lower().split())
    reasons = []
    for reason, pattern in INJECTION_PATTERNS:
        tag = f"injection:{reason}"
        if tag not in reasons and pattern.search(normalized):
            reasons.append(tag)
    return reasons
