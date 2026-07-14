"""
utils/guardrails.py
Bonus feature C: Prompt Injection Detection
Bonus feature D: Output Filtering / Guardrails
"""

from __future__ import annotations
import re

# ── Injection patterns ────────────────────────────────────────────────────────
INJECTION_PATTERNS = [
    # Classic instruction override attempts
    r"ignore (all |previous |your )?(instructions|rules|system|context)",
    r"forget (everything|all|your|previous)",
    r"you are now",
    r"act as (a |an )?(different|new|evil|uncensored|dan|jailbreak)",
    r"pretend (you are|to be)",
    r"(bypass|override|disable) (your |the )?(safety|filter|guardrail|restriction)",
    r"jailbreak",
    r"do anything now",
    r"dan mode",
    r"(system|developer|admin) (mode|override|access)",
    r"</?(system|instructions?)>",
    r"\[INST\]|\[\/INST\]",
    r"<<SYS>>|<</SYS>>",
    # Prompt leaking
    r"(reveal|show|print|repeat|output) (your |the )?(system |initial |original )?(prompt|instructions|rules)",
    r"what (are|were) your instructions",
    # Role manipulation
    r"you('re| are) (a |an )?(human|person|girl|boy|robot|gpt|llm|chatgpt)",
    r"(switch|change) (to |into )?(character|persona|mode)",
]

INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# ── Output safety patterns ────────────────────────────────────────────────────
PII_PATTERNS = {
    "email": re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "api_key": re.compile(r"\b(sk|pk|rk|api)[-_][a-zA-Z0-9]{16,}\b", re.IGNORECASE),
    "phone_eg": re.compile(r"\b01[0-25]\d{8}\b"),   # Egyptian phone numbers
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

HARMFUL_TOPICS = [
    r"how to (make|create|build) (a |an )?(bomb|weapon|exploit|virus|malware)",
    r"(suicide|self.harm) (method|way|how)",
    r"(buy|sell|trade) (drugs|weapons|firearms)",
]
HARMFUL_REGEX = re.compile("|".join(HARMFUL_TOPICS), re.IGNORECASE)


# ── Public API ────────────────────────────────────────────────────────────────
def check_injection(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, warning_message).
    is_safe=True means the query passed, False means it should be blocked.
    """
    # Check for prompt injection
    match = INJECTION_REGEX.search(query)
    if match:
        return False, f"Potential prompt injection detected: '{match.group()[:50]}'"

    # Check for harmful topics
    harm_match = HARMFUL_REGEX.search(query)
    if harm_match:
        return False, f"Query contains harmful content: '{harm_match.group()[:50]}'"

    # Check for suspiciously long queries (potential token flooding)
    if len(query) > 5000:
        return False, "Query exceeds maximum length (5000 characters)."

    return True, ""


def filter_output(text: str) -> str:
    """Redact PII from model output before showing to user."""
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{label.upper()}_REDACTED]", text)
    return text


def validate_sql(sql: str) -> tuple[bool, str]:
    """Block dangerous SQL operations."""
    dangerous = re.compile(
        r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|GRANT|REVOKE|EXEC)\b",
        re.IGNORECASE,
    )
    match = dangerous.search(sql)
    if match:
        return False, f"Blocked SQL operation: {match.group()}"
    return True, ""


def sanitize_filename(name: str) -> str:
    """Remove path traversal and special characters from filenames."""
    name = re.sub(r"[^\w\.\-]", "_", name)
    name = re.sub(r"\.\.", "", name)
    return name[:200]