"""
Layer 3 — Reasoning pipeline.

Two components as described in Architecture.md:

  1. Risk Router   — second escalation gate.  If Layer 1 produced
                     risk_score >= RISK_THRESHOLD or flagged an injection,
                     the LLM is never called; a canned escalation is returned.
                     This is intentional: high-risk tickets (fraud, identity
                     theft, security breach) must never reach a generative model
                     that could hallucinate policy or give bad advice.

  2. Prompt Builder + LLM Call
                   — assembles a domain-specific system prompt, the top-5
                     retrieved corpus excerpts as grounded context, 2-3
                     few-shot examples from sample_support_tickets.csv, and
                     a strict JSON output schema.  Returns all five required
                     output fields via structured output (json_object mode).

Called by the pipeline as:
    from reasoner import reason
    result = reason(preprocess_result, retrieved_chunks)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RISK_THRESHOLD = 1      # risk_score >= this triggers escalation without LLM

SAMPLE_CSV = Path(__file__).parent.parent / "support_tickets" / "sample_support_tickets.csv"

_ESCALATION_MSG = (
    "Thank you for contacting support. Your request requires review by a specialist "
    "due to its sensitive or urgent nature. Our team will follow up with you shortly."
)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class ReasonResult:
    status:        str   # "replied" | "escalated"
    product_area:  str
    response:      str
    justification: str
    request_type:  str   # "product_issue" | "feature_request" | "bug" | "invalid"
    chunk_id:      int = 0  # 1-based index of the excerpt that grounded the response (0 = unknown)


# ---------------------------------------------------------------------------
# 1. Risk Router
# ---------------------------------------------------------------------------
def _should_escalate(pre) -> bool:
    return pre.is_injection or pre.risk_score >= RISK_THRESHOLD


def _make_escalation(pre) -> ReasonResult:
    if pre.is_injection:
        justification = "Prompt or command injection detected; request rejected for safety."
        product_area  = "security"
        request_type  = "invalid"
    else:
        justification = (
            f"Risk score {pre.risk_score:.2f} >= threshold {RISK_THRESHOLD}. "
            f"{pre.risk_reason}"
        )
        product_area = (
            pre.domain.lower()
            if pre.domain not in ("ambiguous", "")
            else "general_support"
        )
        request_type = pre.request_type_hint

    return ReasonResult(
        status="escalated",
        product_area=product_area,
        response=_ESCALATION_MSG,
        justification=justification,
        request_type=request_type,
    )


# ---------------------------------------------------------------------------
# 2a. Few-shot loader
# ---------------------------------------------------------------------------
_FEW_SHOT_CACHE: Optional[list[dict]] = None

# Hand-picked representative justifications for sample examples
_JUSTIFICATIONS = {
    "product_issue": "The corpus article on this topic directly answers the question.",
    "bug":           "No corpus information available to diagnose the outage; escalated to a specialist.",
    "invalid":       "The request is outside the supported domains; responded with an out-of-scope message.",
    "feature_request": "Acknowledged the feature request and noted it for the product team.",
}


def _load_few_shot() -> list[dict]:
    """
    Load 3 diverse examples from sample_support_tickets.csv:
      one product_issue (replied), one bug/escalated, one invalid.
    Results are cached after the first call.
    """
    global _FEW_SHOT_CACHE
    if _FEW_SHOT_CACHE is not None:
        return _FEW_SHOT_CACHE

    if not SAMPLE_CSV.exists():
        _FEW_SHOT_CACHE = []
        return []

    rows: list[dict] = []
    with open(SAMPLE_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    targets = [
        ("product_issue", "Replied"),
        ("bug",           "Escalated"),
        ("invalid",       "Replied"),
    ]
    examples: list[dict] = []
    seen: set[str] = set()

    for want_rt, want_st in targets:
        for row in rows:
            rt = row.get("Request Type", "").strip().lower()
            st = row.get("Status", "").strip().lower()
            if rt == want_rt and st == want_st and rt not in seen:
                resp = row.get("Response", "").replace("\n", " ").strip()
                examples.append({
                    "issue":        row.get("Issue", "").strip()[:300],
                    "company":      row.get("Company", "").strip(),
                    "status":       st,
                    "product_area": row.get("Product Area", "").strip(),
                    "response":     resp[:300] + ("…" if len(resp) > 300 else ""),
                    "justification": _JUSTIFICATIONS.get(rt, "Answered from corpus."),
                    "request_type": rt,
                })
                seen.add(rt)
                break

    _FEW_SHOT_CACHE = examples
    return examples


def _render_few_shot(examples: list[dict]) -> str:
    parts: list[str] = []
    for idx, ex in enumerate(examples, 1):
        output = {
            "status":        ex["status"],
            "product_area":  ex["product_area"],
            "response":      ex["response"],
            "justification": ex["justification"],
            "request_type":  ex["request_type"],
        }
        parts.append(
            "Example " + str(idx) + ":\n"
            "Ticket: " + ex["issue"] + "\n"
            "Company: " + ex["company"] + "\n"
            "Output:\n" + json.dumps(output, indent=2)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2b. Domain context snippets
# ---------------------------------------------------------------------------
_DOMAIN_CTX: dict[str, str] = {
    "HackerRank": (
        "You are a support agent for HackerRank, a technical hiring and assessment platform. "
        "Help recruiters, hiring managers, and candidates with tests, invitations, results, "
        "proctoring, question banks, and platform settings."
    ),
    "Claude": (
        "You are a support agent for Claude, Anthropic's AI assistant. "
        "Help users with subscription plans, API usage, conversations, privacy settings, "
        "account management, and Claude features."
    ),
    "Visa": (
        "You are a support agent for Visa, the global payment network. "
        "Help cardholders and merchants with card services, payments, chargebacks, "
        "travel money, and Visa global assistance."
    ),
    "ambiguous": (
        "You are a multi-domain support agent covering HackerRank, Claude, and Visa. "
        "Identify the relevant domain from context and respond accordingly."
    ),
}


# ---------------------------------------------------------------------------
# 2c. Prompt builder
# ---------------------------------------------------------------------------
# The output schema is written as a plain string (not an f-string) so that
# the JSON braces are never misinterpreted by Python's str.format().
_SCHEMA_BLOCK = """\
Return ONLY a valid JSON object with exactly these six keys — no extra text:
{
  "status"        : "replied" or "escalated",
  "product_area"  : "<the subdirectory name from the source path of the excerpt you used most, e.g. 'managing-tests', 'account-management', 'travel-support'>",
  "response"      : "<user-facing answer grounded in the corpus excerpts — never fabricate facts>",
  "justification" : "<1-2 sentences explaining which corpus excerpt supports the response, or why it was escalated>",
  "request_type"  : "product_issue" or "feature_request" or "bug" or "invalid",
  "chunk_id"      : <integer 1-N: the Excerpt number whose content most directly informed your response; 0 if none>
}"""

_RULES = """\
Rules you must follow:
1. Ground every factual claim in the provided corpus excerpts.  Do NOT use outside knowledge.
2. If the excerpts do not contain enough information to answer safely, set status="escalated".
3. Never fabricate phone numbers, URLs, steps, or policies absent from the excerpts.
4. If the ticket is out of scope (unrelated to supported domains), set request_type="invalid",
   status="replied", and respond politely that the request is outside your scope.
5. Keep the response concise, professional, and user-facing."""


def _build_context_block(chunks) -> str:
    if not chunks:
        return "[No relevant corpus excerpts found for this ticket.]"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Excerpt {i} | {c.source}]\n{c.text}")
    return "\n\n".join(parts)


def _build_prompt(pre, chunks) -> tuple[str, str]:
    """Return (system_prompt, user_message)."""
    domain_ctx = _DOMAIN_CTX.get(pre.domain, _DOMAIN_CTX["ambiguous"])
    few_shots  = _load_few_shot()

    system = "\n\n".join([
        domain_ctx,
        _RULES,
        _SCHEMA_BLOCK,
        "--- FEW-SHOT EXAMPLES ---\n" + _render_few_shot(few_shots) + "\n--- END EXAMPLES ---",
    ])

    ticket_text = (pre.english_text or pre.full_text).strip()
    context_block = _build_context_block(chunks)

    user = (
        "Company: "       + pre.domain + "\n"
        "Intent hint: "   + pre.request_type_hint + "\n\n"
        "Ticket:\n"       + ticket_text + "\n\n"
        "--- CORPUS EXCERPTS ---\n" + context_block + "\n--- END CORPUS EXCERPTS ---\n\n"
        "Respond with the JSON object only."
    )

    return system, user


# ---------------------------------------------------------------------------
# 3. Schema coercion
# ---------------------------------------------------------------------------
_VALID_STATUS = {"replied", "escalated"}
_VALID_RT     = {"product_issue", "feature_request", "bug", "invalid"}


def _coerce(raw: dict, pre) -> ReasonResult:
    """Enforce valid enum values; substitute safe defaults if the LLM drifts."""
    status = str(raw.get("status", "escalated")).lower().strip()
    if status not in _VALID_STATUS:
        status = "escalated"

    request_type = str(raw.get("request_type", pre.request_type_hint)).lower().strip()
    if request_type not in _VALID_RT:
        request_type = pre.request_type_hint

    product_area  = str(raw.get("product_area", pre.domain.lower())).strip() or pre.domain.lower()
    response      = str(raw.get("response", _ESCALATION_MSG)).strip()      or _ESCALATION_MSG
    justification = str(raw.get("justification", "")).strip()

    # Extract chunk_id — the LLM tells us which excerpt grounded its response
    try:
        chunk_id = int(raw.get("chunk_id", 0))
        if chunk_id < 0:
            chunk_id = 0
    except (TypeError, ValueError):
        chunk_id = 0

    return ReasonResult(
        status=status,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=request_type,
        chunk_id=chunk_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reason(pre, chunks) -> ReasonResult:
    """
    Full Layer 3 pipeline for a single ticket.

    Args:
        pre:    PreprocessResult from Layer 1.
        chunks: list[RetrievedChunk] from Layer 2 (may be empty).

    Returns:
        ReasonResult with all five output fields populated.
    """
    from llm_client import call_llm_json

    # Gate 2 — risk router
    if _should_escalate(pre):
        return _make_escalation(pre)

    # Build prompt
    system, user = _build_prompt(pre, chunks)

    # LLM call
    try:
        raw = call_llm_json(
            messages=[{"role": "user", "content": user}],
            system=system,
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as exc:
        return ReasonResult(
            status="escalated",
            product_area=pre.domain.lower() if pre.domain != "ambiguous" else "general_support",
            response=_ESCALATION_MSG,
            justification=f"LLM call failed: {exc}",
            request_type=pre.request_type_hint,
        )

    return _coerce(raw, pre)
