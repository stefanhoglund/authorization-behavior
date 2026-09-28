# parsing.py

import re
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


def parse_semantic_decision(
    text: str,
) -> str | None:
    text = text.strip()

    # 1. Exact benchmark format
    if text in {"ALLOW", "DENY"}:
        return text

    candidates = []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return None

    # 2. Decision at beginning of response:
    #    DENY
    #    DENY (...)
    #    ALLOW — explanation
    first_match = re.match(
        r"^(ALLOW|DENY)\b",
        lines[0],
        flags=re.IGNORECASE,
    )

    if first_match:
        candidates.append(first_match.group(1).upper())

    # 3. Isolated decision on final line,
    #    common for Phi-4.
    if lines[-1].upper() in {
        "ALLOW",
        "DENY",
    }:
        candidates.append(lines[-1].upper())

    # No sufficiently explicit decision found.
    if not candidates:
        return None

    # Refuse to parse contradictory signals.
    if len(set(candidates)) > 1:
        return None

    return candidates[0]
