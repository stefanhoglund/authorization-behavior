# parsing.py

from enum import Enum


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"
    FORMAT_ERROR = "FORMAT_ERROR"


def parse_decision(text: str) -> Decision:
    normalized = text.strip().upper()

    if normalized == "ALLOW":
        return Decision.ALLOW

    if normalized == "DENY":
        return Decision.DENY

    if normalized == "ESCALATE":
        return Decision.ESCALATE

    return Decision.FORMAT_ERROR