"""
LLM backend wrapper for StateGen using litellm.

Supports two providers:
  - Together.ai  (backend="together",   default)
  - ThetaEdge    (backend="thetaedge")

API keys and the ThetaEdge base URL are loaded from a .env file at the project root.
Together.ai calls are retried up to 3 times with exponential backoff on rate-limit or
timeout errors; ThetaEdge fails fast with no retry.
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv
import litellm

load_dotenv()

TOGETHER_MODEL: str = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
THETAEDGE_MODEL: str = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPSEEK_MODEL: str = "openai/deepseek-coder"


_MAX_RETRIES: int = 3
_RETRY_DELAYS: list[float] = [1.0, 2.0, 4.0]  # seconds before each successive retry


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a rate-limit or timeout error."""
    msg = str(exc).lower()
    return any(token in msg for token in ("rate", "timeout", "429", "524", "too many"))


def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    backend: str = "together",
) -> str:
    """
    Call the specified LLM backend and return the generated text.

    Args:
        prompt:        The user-facing prompt.
        system_prompt: Optional system instruction prepended to the conversation.
        temperature:   Sampling temperature (default 0.2 for deterministic code gen).
        max_tokens:    Maximum tokens to generate.
        backend:       Provider to use — "together" (default) or "thetaedge".

    Returns:
        The raw response content string from the model.

    Raises:
        ValueError:   If an unsupported backend name is given.
        RuntimeError: If Together.ai fails after all retries.
        Exception:    Any non-retryable litellm / provider error propagates directly.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if backend == "together":
        return _call_together(messages, temperature, max_tokens)
    elif backend == "thetaedge":
        return _call_thetaedge(messages, temperature, max_tokens)
    elif backend == "deepseek":
        return _call_deepseek(messages, temperature, max_tokens)
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Valid choices are 'together' or 'thetaedge'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_together(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Together.ai with retry logic for rate-limit / timeout errors."""
    api_key: Optional[str] = os.getenv("TOGETHER_API_KEY")
    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):  # attempts: 0, 1, 2, 3
        try:
            response = litellm.completion(
                model=TOGETHER_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if not _is_retryable(exc):
                raise  # propagate non-retryable errors immediately
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                print(
                    f"[llm_backend] Together.ai attempt {attempt + 1} failed "
                    f"({type(exc).__name__}). Retrying in {delay}s..."
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Together.ai call failed after {_MAX_RETRIES} retries. "
        f"Last error: {last_exc}"
    )


def _call_thetaedge(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call ThetaEdge with no retry (fail fast)."""
    api_key: Optional[str] = os.getenv("THETAEDGE_API_KEY")
    base_url: Optional[str] = os.getenv("THETAEDGE_BASE_URL")

    response = litellm.completion(
        model=THETAEDGE_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=base_url,
    )
    return response.choices[0].message.content

def _call_deepseek(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call DeepSeek API (OpenAI-compatible, no retry)."""
    api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    base_url: Optional[str] = os.getenv("DEEPSEEK_BASE_URL")

    response = litellm.completion(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=base_url,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== llm_backend.py smoke test (Together.ai) ===\n")
    output = generate(
        prompt="Write a hello world function in Python.",
        system_prompt="You are a helpful Python programming assistant. Be concise.",
    )
    print(output)
