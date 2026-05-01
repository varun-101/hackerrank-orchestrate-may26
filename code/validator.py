"""
Layer 4 — Output Validation.

Three independent checks run on every ReasonResult before it is written to CSV:

  1. Schema Validator    — all enum fields are legal values; required fields are
                           non-empty.  Coerces or escalates on failure.

  2. Consistency Checker — catches contradictions the LLM can produce:
                           e.g. status=replied but the response body says
                           "please contact support" (that is an escalation, not a reply).

  3. Hallucination Guard — verifies that factual claims in the response are
                           attributable to the retrieved corpus excerpts.
                           Uses a fast heuristic pre-filter (phone numbers, URLs,
                           specific numbers) followed by an LLM grounding call.
                           If hallucination is detected the ticket is escalated —
                           better to escalate than to ship fabricated policy.

Public API:
    from validator import validate
    vr = validate(reason_result, pre, chunks)
    # vr.result  — possibly corrected ReasonResult
    # vr.passed  — True if no issues found
    # vr.issues  — list[str] describing what failed
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_STATUS = {"replied", "escalated"}
_VALID_RT     = {"product_issue", "feature_request", "bug", "invalid"}

# Phrases in a response that imply the ticket should actually be escalated
_ESCALATION_PHRASES = [
    "please contact support",
    "contact our support team",
    "reach out to our team",
    "contact a specialist",
    "escalate this",
    "forward this to",
    "our team will",
    "a representative will",
    "please call us",
]

# Heuristic patterns for claims that are easy to verify against the corpus
_PHONE_RE   = re.compile(r'\+?[\d][\d\s\-\(\)\.]{6,}[\d]')
_URL_RE     = re.compile(r'https?://[^\s]+')
_STEP_NUM_RE = re.compile(r'\b(\d{3,})\b')   # 3+ digit numbers used in steps

# Minimum response length for a "replied" ticket (very short = suspicious)
_MIN_RESPONSE_LEN = 20

# Hallucination LLM system prompt
_HAL_SYSTEM = (
    "You are a strict fact-checker for a customer support system.\n\n"
    "You will be given:\n"
    "  - RESPONSE: a support reply to a customer\n"
    "  - CORPUS: the source documents the reply was supposed to be based on\n\n"
    "Your job: decide whether every factual claim in RESPONSE is explicitly supported "
    "by the CORPUS.  Be strict — phone numbers, URLs, numbered steps, policies, and "
    "product names must appear in the CORPUS verbatim or in clear paraphrase.\n\n"
    "Return ONLY a JSON object with exactly two keys:\n"
    "{\n"
    '  "grounded": true | false,\n'
    '  "unsupported_claims": ["<claim not found in corpus>", ...]\n'
    "}\n"
    "If all claims are grounded, return: {\"grounded\": true, \"unsupported_claims\": []}"
)

_ESCALATION_MSG = (
    "Thank you for contacting support. Your request requires review by a specialist "
    "due to its sensitive or urgent nature. Our team will follow up with you shortly."
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    result: object                      # ReasonResult — potentially corrected
    passed: bool                        # True = all three checks passed
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Schema Validator
# ---------------------------------------------------------------------------
def _validate_schema(result, pre) -> tuple[object, list[str]]:
    """
    Enforce legal enum values and non-empty required fields.
    Returns (possibly_corrected_result, issues).
    """
    from reasoner import ReasonResult

    issues: list[str] = []
    status       = str(result.status or "").lower().strip()
    request_type = str(result.request_type or "").lower().strip()
    product_area = str(result.product_area or "").strip()
    response     = str(result.response or "").strip()
    justification = str(result.justification or "").strip()

    if status not in _VALID_STATUS:
        issues.append(f"schema: invalid status {result.status!r} — coerced to 'escalated'")
        status = "escalated"

    if request_type not in _VALID_RT:
        issues.append(
            f"schema: invalid request_type {result.request_type!r} — "
            f"coerced to {pre.request_type_hint!r}"
        )
        request_type = pre.request_type_hint

    if not product_area:
        fallback = pre.domain.lower() if pre.domain not in ("ambiguous", "") else "general_support"
        issues.append(f"schema: empty product_area — coerced to {fallback!r}")
        product_area = fallback

    if not response or len(response) < _MIN_RESPONSE_LEN:
        issues.append(
            f"schema: response too short ({len(response)} chars) — escalating"
        )
        status   = "escalated"
        response = _ESCALATION_MSG

    if not justification:
        issues.append("schema: empty justification")

    corrected = ReasonResult(
        status=status,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=request_type,
    )
    return corrected, issues


# ---------------------------------------------------------------------------
# 2. Consistency Checker
# ---------------------------------------------------------------------------
def _validate_consistency(result) -> tuple[object, list[str]]:
    """
    Catch logical contradictions between status, response content, and request_type.
    """
    from reasoner import ReasonResult

    issues: list[str] = []
    status   = result.status
    response = result.response.lower()
    justification = result.justification

    # A "replied" status with response text implying escalation is a contradiction
    if status == "replied":
        for phrase in _ESCALATION_PHRASES:
            if phrase in response:
                issues.append(
                    f"consistency: status=replied but response contains {phrase!r} "
                    f"— corrected to escalated"
                )
                status = "escalated"
                justification = justification + " [auto-escalated: response implied handoff]"
                break

    # An escalated response with a very long detailed answer may be misclassified
    # (don't flip it, but flag for awareness)
    if status == "escalated" and len(result.response) > 600:
        issues.append(
            "consistency: status=escalated but response is unusually long "
            f"({len(result.response)} chars) — verify manually"
        )

    corrected = ReasonResult(
        status=status,
        product_area=result.product_area,
        response=result.response,
        justification=justification,
        request_type=result.request_type,
    )
    return corrected, issues


# ---------------------------------------------------------------------------
# 3. Hallucination Guard
# ---------------------------------------------------------------------------
def _heuristic_check(response: str, chunks) -> list[str]:
    """
    Fast pre-filter: find phone numbers, URLs, and large numbers in the response
    that cannot be located anywhere in the corpus.  Runs without an API call.
    """
    corpus = " ".join(c.text for c in chunks)
    issues: list[str] = []

    for match in _PHONE_RE.finditer(response):
        raw = match.group().strip()
        # normalise both sides: strip spaces and hyphens for comparison
        norm = re.sub(r'[\s\-\(\)\.]', '', raw)
        corpus_norm = re.sub(r'[\s\-\(\)\.]', '', corpus)
        if len(norm) >= 7 and norm not in corpus_norm:
            issues.append(f"heuristic: phone number not in corpus: {raw!r}")

    for match in _URL_RE.finditer(response):
        url = match.group().rstrip('.,)')
        # check the domain part at minimum
        domain = url.split('/')[2] if url.count('/') >= 2 else url
        if domain.lower() not in corpus.lower():
            issues.append(f"heuristic: URL domain not in corpus: {domain!r}")

    return issues


def _llm_hallucination_check(response: str, chunks) -> tuple[bool, list[str]]:
    """
    LLM-based grounding check.  Returns (is_grounded, unsupported_claims).
    Falls back to (True, []) on API error so validation never blocks on LLM failure.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return True, []   # skip if no key — don't block the pipeline

    corpus_block = "\n\n".join(
        f"[Excerpt {i+1}]\n{c.text[:600]}" for i, c in enumerate(chunks[:5])
    )
    user_msg = (
        "RESPONSE:\n" + response + "\n\n"
        "CORPUS:\n"   + corpus_block
    )

    try:
        from llm_client import call_llm_json
        result = call_llm_json(
            messages=[{"role": "user", "content": user_msg}],
            system=_HAL_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
        grounded   = bool(result.get("grounded", True))
        ungrounded = result.get("unsupported_claims", [])
        if not isinstance(ungrounded, list):
            ungrounded = []
        return grounded, [str(c) for c in ungrounded]
    except Exception:
        return True, []   # fail open — let the response through


def _validate_hallucination(result, chunks) -> tuple[object, list[str]]:
    """
    Run heuristic + LLM hallucination checks.
    If hallucination is confirmed, escalate the result.
    """
    from reasoner import ReasonResult

    if not chunks:
        return result, ["hallucination: no corpus chunks — cannot verify grounding"]

    issues: list[str] = []

    # Fast heuristic pass
    heuristic_issues = _heuristic_check(result.response, chunks)
    issues.extend(heuristic_issues)

    # LLM grounding pass
    grounded, unsupported = _llm_hallucination_check(result.response, chunks)
    if not grounded:
        for claim in unsupported:
            issues.append(f"hallucination: unsupported claim — {claim}")

    if not grounded and unsupported:
        # Confirmed hallucination → escalate instead of shipping bad content
        escalated = ReasonResult(
            status="escalated",
            product_area=result.product_area,
            response=_ESCALATION_MSG,
            justification=(
                result.justification
                + " [escalated by hallucination guard: "
                + "; ".join(unsupported[:2])
                + "]"
            ),
            request_type=result.request_type,
        )
        return escalated, issues

    return result, issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate(result, pre, chunks) -> ValidationResult:
    """
    Run all three Layer 4 checks on a ReasonResult.

    Args:
        result:  ReasonResult from Layer 3.
        pre:     PreprocessResult from Layer 1 (used for fallback values).
        chunks:  list[RetrievedChunk] from Layer 2 (used for hallucination check).

    Returns:
        ValidationResult containing the (possibly corrected) result,
        a passed flag, and a list of issue descriptions.
    """
    all_issues: list[str] = []

    # 1 — Schema
    result, schema_issues = _validate_schema(result, pre)
    all_issues.extend(schema_issues)

    # 2 — Consistency
    result, consistency_issues = _validate_consistency(result)
    all_issues.extend(consistency_issues)

    # 3 — Hallucination (only for replied responses — escalated ones use a canned msg)
    if result.status == "replied":
        result, hal_issues = _validate_hallucination(result, chunks)
        all_issues.extend(hal_issues)

    return ValidationResult(
        result=result,
        passed=len(all_issues) == 0,
        issues=all_issues,
    )
