"""
Shared LLM client — thin wrapper around DeepSeek's OpenAI-compatible API.

All pipeline layers import from here so the model, endpoint, and retry
logic live in one place.  To swap the provider, change _BASE_URL and
DEFAULT_MODEL; nothing in the pipeline layers needs updating.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

DEFAULT_MODEL = "deepseek-chat"
_BASE_URL = "https://api.deepseek.com"
_MAX_RETRIES = 2
_RETRY_DELAY = 1.5  # seconds; doubled on each retry

_client_instance: Optional[OpenAI] = None


def _client() -> OpenAI:
    """Return (or lazily create) the shared DeepSeek client."""
    global _client_instance
    if _client_instance is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client_instance = OpenAI(api_key=api_key, base_url=_BASE_URL)
    return _client_instance


def call_llm(
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Call the LLM and return its text response.

    Args:
        messages:    List of {"role": ..., "content": ...} dicts (no system msg).
        system:      Optional system prompt prepended to the message list.
        model:       DeepSeek model name (default: deepseek-chat).
        max_tokens:  Max tokens in the response.
        temperature: Sampling temperature; 0.0 = deterministic.

    Returns:
        The model's text response as a string.

    Raises:
        EnvironmentError: If DEEPSEEK_API_KEY is not set.
        openai.APIError:  If all retries are exhausted.
    """
    all_messages = _build_messages(system, messages)
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=model,
                messages=all_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError) as e:
            last_exc = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except APIError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    raise last_exc


def call_llm_json(
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call the LLM in JSON mode and return the parsed response dict.

    The system prompt MUST instruct the model to return valid JSON, otherwise
    DeepSeek may refuse JSON mode.  Use call_llm() for free-text responses.

    Returns:
        Parsed dict from the model's JSON response.

    Raises:
        ValueError:  If the response cannot be parsed as JSON after retries.
        EnvironmentError / openai.APIError: same as call_llm().
    """
    all_messages = _build_messages(system, messages)
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=model,
                messages=all_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            return json.loads(text)
        except (RateLimitError, APIConnectionError) as e:
            last_exc = e
            time.sleep(_RETRY_DELAY * (attempt + 1))
        except APIError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
        except json.JSONDecodeError as e:
            # Malformed JSON from the model — wrap and raise immediately
            raise ValueError(f"LLM returned non-JSON: {text!r}") from e

    raise last_exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_messages(
    system: Optional[str],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}] + messages
    return list(messages)
