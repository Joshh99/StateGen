import os
import time
import logging
from typing import Optional

from dotenv import load_dotenv
import litellm
from litellm.exceptions import ServiceUnavailableError

from config import LLM_MODEL, LLM_BASE_URL, MAX_NEW_TOKENS, TEMPERATURE

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_DEFAULT = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)

_DEFAULT_MODELS = {
    "together_ai":       "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "thetaedge":         "openai/Qwen/Qwen2.5-Coder-32B-Instruct",
    "openai_compatible": "deepseek-coder",
    "openai":            "gpt-4o",
    "anthropic":         "claude-sonnet-4-20250514",
    "azure":             "gpt-4o-stategen",
    "gemini":            "gemini/gemini-2.5-flash",
}

_MAX_RETRIES = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a rate-limit or timeout error."""
    if isinstance(exc, ServiceUnavailableError):
        return True
    msg = str(exc).lower()
    return any(token in msg for token in ("rate", "timeout", "429", "524", "too many"))


class LLMAgent:
    """
    LiteLLM-based LLM agent with token tracking and latency recording.

    Providers:
        "together_ai"       — Together.ai  (default: Meta-Llama-3.1-8B-Instruct-Turbo)
        "thetaedge"         — ThetaEdge    (Qwen2.5-Coder-32B-Instruct)
        "openai_compatible" — DeepSeek or any OpenAI-compatible endpoint
        "openai"            — OpenAI API   (GPT-4o, etc.)
        "anthropic"         — Anthropic    (Claude)

    Usage:
        agent = LLMAgent(provider="together_ai", tracker=tracker)
        code  = agent.generate(prompt, task_id="BigCodeBench/0",
                               method="self_planning", call_type="generation")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tracker=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
    ):
        self.provider    = provider or os.getenv("LLM_PROVIDER", "together_ai")
        self.model       = model or _DEFAULT_MODELS.get(self.provider, LLM_MODEL)
        self.tracker     = tracker
        self.temperature = temperature
        self.max_tokens  = max_tokens

        # --- API key resolution ---
        if api_key:
            self._api_key = api_key
        elif self.provider == "together_ai":
            self._api_key = os.getenv("TOGETHER_API_KEY")
        elif self.provider == "thetaedge":
            self._api_key = os.getenv("THETAEDGE_API_KEY")
        elif self.provider == "anthropic":
            self._api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "openai_compatible":
            self._api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif self.provider == "azure":
            self._api_key = os.getenv("AZURE_API_KEY")
        elif self.provider == "gemini":
            self._api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        else:
            self._api_key = os.getenv("OPENAI_API_KEY")

        # --- Base URL resolution ---
        if base_url:
            self._base_url = base_url
        elif self.provider == "thetaedge":
            self._base_url = os.getenv("THETAEDGE_BASE_URL")
        elif self.provider == "openai_compatible":
            self._base_url = os.getenv("LLM_BASE_URL", LLM_BASE_URL)
        elif self.provider == "azure":
            self._base_url = os.getenv("AZURE_API_BASE", "https://stategen.openai.azure.com/")
        else:
            self._base_url = None

        # Azure API version (only used when provider == "azure")
        self._api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

        self._last_latency_ms: Optional[float] = None
        self._latency_log: list[dict] = []

        logger.info(f"LLMAgent ready: {self.provider} / {self.model}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE (used by all baselines)
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        task_id: str = "",
        method: str = "",
        call_type: str = "generation",
        attempt: int = 0,
    ) -> str:
        """
        Generate text from a prompt via LiteLLM.

        Args:
            prompt:     User message / instruction.
            system:     Optional system prompt override.
            task_id:    For token tracking (e.g. "BigCodeBench/42").
            method:     Method name for tracking (e.g. "self_planning").
            call_type:  Stage label: "generation" | "planning" | "explanation" | "repair".
            attempt:    Retry index within a task (0 = first attempt).

        Returns:
            Generated text string.
        """
        messages = []
        if system or SYSTEM_DEFAULT:
            messages.append({"role": "system", "content": system or SYSTEM_DEFAULT})
        messages.append({"role": "user", "content": prompt})

        response = self._call_with_retry(messages)

        text = response.choices[0].message.content or ""

        input_tokens  = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        latency_ms = getattr(response, "_response_ms", None)
        self._last_latency_ms = latency_ms
        self._latency_log.append({
            "task_id":    task_id,
            "method":     method,
            "call_type":  call_type,
            "attempt":    attempt,
            "latency_ms": latency_ms,
            "timestamp":  time.time(),
        })

        if self.tracker and task_id:
            self.tracker.record(
                task_id=task_id,
                method=method,
                call_type=call_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempt=attempt,
            )

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # LITELLM CALL WITH RETRY (ported from src/llm_backend.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _call_with_retry(self, messages: list[dict]):
        """
        Call litellm.completion() with exponential backoff on retryable errors.

        Together.ai and OpenAI-compatible providers get retries.
        ThetaEdge fails fast (no retry) — same behaviour as src/llm_backend.py.
        """
        PROVIDER_PREFIXES = {
            "together_ai": "together_ai/",
            "openai_compatible": "",
            "openai": "",
            "anthropic": "",
            "azure": "azure/",
            "gemini": "",
        }
        prefix = PROVIDER_PREFIXES.get(self.provider, "")
        model_str = f"{prefix}{self.model}" if prefix and not self.model.startswith(prefix) else self.model

        max_attempts = 1 if self.provider == "thetaedge" else _MAX_RETRIES + 1
        last_exc = None

        for attempt in range(max_attempts):
            try:
                kwargs = {
                    "model":       model_str,
                    "messages":    messages,
                    "temperature": self.temperature,
                    "max_tokens":  self.max_tokens,
                    "api_key":     self._api_key,
                }
                if self._base_url:
                    kwargs["api_base"] = self._base_url
                if self.provider == "azure":
                    kwargs["api_version"] = self._api_version

                return litellm.completion(**kwargs)

            except Exception as exc:
                if not _is_retryable(exc) or self.provider == "thetaedge":
                    raise
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        f"[LLMAgent] {self.provider} attempt {attempt + 1} failed "
                        f"({type(exc).__name__}). Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"{self.provider} call failed after {_MAX_RETRIES} retries. "
            f"Last error: {last_exc}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LATENCY ACCESS
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def last_latency_ms(self) -> Optional[float]:
        """Latency of the most recent LLM call in milliseconds."""
        return self._last_latency_ms

    @property
    def latency_log(self) -> list[dict]:
        """Full log of per-call latencies."""
        return list(self._latency_log)
