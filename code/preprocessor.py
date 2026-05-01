"""
Layer 1 — Pre-processing pipeline.

Runs four sub-components in sequence before any retrieval or reasoning:

  1. Guard Rails      — prompt/command injection detection (regex, multilingual).
                        Must stay regex: using an LLM here would let injections
                        manipulate their own safety check.
  2. Language Detect  — ISO-639-1 code via langdetect (fast, no API cost).
  3. English Normalise— translate non-English text to English via DeepSeek
                        so retrieval and classification work on a uniform input.
  4. LLM Classify     — single DeepSeek call that returns domain, request_type,
                        risk_score, and risk_reason.  Replaces regex heuristics
                        because an LLM handles edge cases that patterns miss.

Guard rails short-circuit before the LLM classify step; injected tickets
are never forwarded to the generative model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT = True
except ImportError:
    _LANGDETECT = False

# ---------------------------------------------------------------------------
# Guard Rails — multilingual injection + command patterns
# (intentionally regex-only — must not depend on the LLM being guarded)
# ---------------------------------------------------------------------------

_INJECTION_EN: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"forget\s+(everything|all)\s+(you|i)",
    r"show\s+(me\s+)?your\s+(system\s+)?prompt",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"what\s+are\s+your\s+(rules|instructions|guidelines)",
    r"dump\s+(your\s+)?(internal\s+)?(rules|logic|documents|context)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"act\s+as\s+(if\s+you|a|an)\b",
    r"pretend\s+(you|to\s+be)",
    r"\bDAN\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"override\s+(your\s+)?(safety|content)\s+(filter|policy|rules)",
    r"you\s+have\s+no\s+(restrictions|limits|rules)\b",
    r"bypass\s+(your\s+)?(safety|content|filter)",
]]

_INJECTION_FR: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore\s+(toutes?\s+les?\s+)?instructions",
    r"affiche\s+(toutes?\s+les?\s+)?règles",
    r"montre\s+(toutes?\s+les?\s+)?règles",
    r"révèle\s+(ton\s+)?prompt",
    r"documents?\s+récupérés",
    r"logique\s+(interne|exacte)",
    r"règles\s+internes",
    r"tu\s+es\s+maintenant\s+(un|une)\b",
    r"fais\s+semblant\s+d",
]]

_INJECTION_ES: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"ignora\s+(todas?\s+las?\s+)?instrucciones",
    r"muestra\s+(todas?\s+las?\s+)?reglas",
    r"revela\s+(tu\s+)?prompt",
    r"eres\s+ahora\s+(un|una)\b",
    r"actúa\s+como\s+si",
    r"olvida\s+(todas?\s+las?\s+)?instrucciones",
]]

_INJECTION_CMD: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"delete\s+all\s+files",
    r"\brm\s+-rf\b",
    r"format\s+(the\s+)?(hard\s+)?drive",
    r"drop\s+(table|database|schema)\b",
    r"execute\s+(this\s+)?(code|command|script)",
    r"run\s+(this\s+)?(code|command|script|shell)",
    r"\bsudo\s+",
    r"eval\s*\(",
    r"os\.system\s*\(",
    r"subprocess\.(call|run|Popen)",
]]

_ALL_INJECTION = _INJECTION_EN + _INJECTION_FR + _INJECTION_ES + _INJECTION_CMD

# ---------------------------------------------------------------------------
# LLM system prompt for the combined domain / intent / risk classification
# ---------------------------------------------------------------------------
_CLASSIFY_SYSTEM = """\
You are a support ticket triage classifier for a multi-domain customer support system.

The three supported domains are:
- HackerRank : coding assessments, candidates, recruiters, tests, hiring platform
- Claude     : Anthropic's AI assistant, claude.ai, API usage, plans, conversations
- Visa       : payment cards, card network, merchants, chargebacks, travel money

Return ONLY a JSON object with exactly these four keys:

{
  "domain"       : "HackerRank" | "Claude" | "Visa" | "ambiguous",
  "request_type" : "product_issue" | "feature_request" | "bug" | "invalid",
  "risk_score"   : <float 0.0–1.0>,
  "risk_reason"  : "<one concise sentence>"
}

--- domain rules ---
Use the company_hint if it names a real domain.
Otherwise infer from ticket content.
Use "ambiguous" for out-of-scope, cross-domain, or unrecognisable tickets.

--- request_type rules ---
"invalid"         : out-of-scope question (celebrities, weather, etc.), pure greeting, spam
"bug"             : something is broken, not working, throwing an error, site is down
"feature_request" : asking for new or additional functionality
"product_issue"   : support question about an existing feature (default when unsure)

--- risk_score rules ---
0.9–1.0 : fraud, stolen card/identity, security breach, hacked account, emergency
0.5–0.8 : account deletion, billing dispute, unauthorised access, PII data request
0.2–0.4 : password reset, personal data change, sensitive account modification
0.0–0.1 : general FAQ, feature request, minor product question

Do not include any text outside the JSON object.\
"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    is_injection: bool
    language: str                       # ISO-639-1 code, e.g. "en", "fr"
    domain: str                         # HackerRank | Claude | Visa | ambiguous
    request_type_hint: str              # product_issue | feature_request | bug | invalid
    risk_score: float                   # 0.0–1.0
    risk_reason: str = ""               # human-readable explanation of the risk score
    risk_flags: list[str] = field(default_factory=list)  # machine-readable markers
    full_text: str = ""                 # raw subject + issue (original language)
    english_text: Optional[str] = None  # English-normalised text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preprocess(issue: str, subject: str = "", company: Optional[str] = None) -> PreprocessResult:
    """
    Full Layer 1 pipeline for a single ticket.

    Steps:
      1. Concatenate subject + issue.
      2. Guard rails check (regex, all languages).
      3. Language detection.
      4. English normalisation via LLM (skipped if already English).
      5. Guard rails re-check on translated text.
      6. LLM classification (domain, intent, risk) — skipped for injections.
    """
    full_text = " ".join(filter(None, [subject, issue])).strip()

    # Step 2 — guard rails on original text
    is_injection, inj_flags = _check_injection(full_text)

    # Step 3 — language detection
    language = _detect_language(full_text)

    # Step 4 — translate to English if needed
    english_text = _normalize_to_english(full_text, language)

    # Step 5 — guard rails on translated text (catches foreign-language injections
    #           that slipped past the multilingual patterns above)
    if not is_injection and english_text and english_text != full_text:
        is_injection_translated, trans_flags = _check_injection(english_text)
        if is_injection_translated:
            is_injection = True
            inj_flags = trans_flags + ["detected_after_translation"]

    # Step 6 — LLM classification (short-circuit for injections)
    classify_text = english_text if english_text else full_text

    if is_injection:
        # Never send injected text to the generative model
        domain = _company_or_ambiguous(company)
        request_type_hint = "invalid"
        risk_score = 0.9
        risk_reason = "Prompt or command injection attempt detected."
        risk_flags = inj_flags
    else:
        classification = _llm_classify(classify_text, company)
        domain = classification.get("domain", _company_or_ambiguous(company))
        request_type_hint = classification.get("request_type", "product_issue")
        risk_score = float(classification.get("risk_score", 0.0))
        risk_reason = classification.get("risk_reason", "")
        risk_flags = []

    return PreprocessResult(
        is_injection=is_injection,
        language=language,
        domain=domain,
        request_type_hint=request_type_hint,
        risk_score=risk_score,
        risk_reason=risk_reason,
        risk_flags=risk_flags,
        full_text=full_text,
        english_text=english_text,
    )


# ---------------------------------------------------------------------------
# Sub-components (private)
# ---------------------------------------------------------------------------
def _check_injection(text: str) -> tuple[bool, list[str]]:
    flags = [p.pattern for p in _ALL_INJECTION if p.search(text)]
    return bool(flags), flags


def _detect_language(text: str) -> str:
    if not _LANGDETECT or not text.strip():
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"


def _normalize_to_english(text: str, language: str) -> Optional[str]:
    """Translate text to English using DeepSeek. Returns None on failure or if already English."""
    if language == "en":
        return text
    if not text.strip():
        return text

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return text  # skip silently when no key

    try:
        from llm_client import call_llm
        return call_llm(
            messages=[{"role": "user", "content": text}],
            system=(
                "Translate the following text to English. "
                "Output ONLY the translated text with no explanation or commentary."
            ),
            max_tokens=512,
            temperature=0.0,
        ).strip()
    except Exception:
        return text  # fall back to original on any error


def _llm_classify(text: str, company: Optional[str]) -> dict:
    """Single DeepSeek call returning domain, request_type, risk_score, risk_reason."""
    company_hint = (
        f"company_hint: {company.strip()}"
        if company and company.strip() not in ("", "None")
        else "company_hint: unknown"
    )
    user_msg = f"{company_hint}\n\nTicket:\n{text}"

    try:
        from llm_client import call_llm_json
        return call_llm_json(
            messages=[{"role": "user", "content": user_msg}],
            system=_CLASSIFY_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
    except Exception:
        return {
            "domain": _company_or_ambiguous(company),
            "request_type": "product_issue",
            "risk_score": 0.0,
            "risk_reason": "Classification unavailable (LLM error).",
        }


def _company_or_ambiguous(company: Optional[str]) -> str:
    if company and company.strip() not in ("", "None"):
        return company.strip()
    return "ambiguous"
