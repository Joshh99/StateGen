# agents/llm_agent.py
# API-based LLM wrapper.
# Supports OpenAI, Anthropic, and any OpenAI-compatible endpoint
# (Together AI, Fireworks, vLLM, etc.).
# Token counts come directly from the API response — no manual counting needed.

import os
import logging
from dataclasses import dataclass
from typing import Optional

from config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE
from core.models import State, ProgramContext

logger = logging.getLogger(__name__)

SYSTEM_DEFAULT = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)


@dataclass
class _GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMAgent:
    """
    Wraps an LLM API and records token usage via TokenTracker.

    Providers:
        "openai"            — OpenAI API (GPT-4o, GPT-4-turbo, etc.)
        "anthropic"         — Anthropic API (Claude)
        "openai_compatible" — Any OpenAI-compatible endpoint
                              (Together AI, Fireworks, local vLLM, etc.)
                              Set base_url= to the endpoint URL.

    Usage:
        agent = LLMAgent(model="gpt-4o", provider="openai", tracker=tracker)
        code  = agent.generate(prompt, task_id="BigCodeBench/0",
                               method="self_planning", call_type="generation")

    StateController compatibility:
        generate_for_state() and generate_decomposition() are preserved
        so the existing StateController works without changes.
    """

    PROVIDERS = ("openai", "anthropic", "openai_compatible", "azure_openai")

    def __init__(
        self,
        model: str = LLM_MODEL,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        tracker=None,                     # core.token_tracker.TokenTracker
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
        request_timeout: int = 120,
    ):
        if provider not in self.PROVIDERS:
            raise ValueError(f"provider must be one of {self.PROVIDERS}, got '{provider}'")

        self.model           = model
        self.provider        = provider
        self.tracker         = tracker
        self.temperature     = temperature
        self.max_tokens      = max_tokens
        self.request_timeout = request_timeout

        # Resolve API key from arg → env var
        if api_key is None:
            if provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openai_compatible":
                api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif provider == "azure_openai":
                api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")

        # Build the client
        if provider == "azure_openai":
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version or "2024-12-01-preview",
                timeout=self.request_timeout,
            )
        elif provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=self.request_timeout)
        else:  # anthropic
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)

        logger.info(f"LLMAgent ready: {provider} / {model}")

    # ─────────────────────────────────────────
    # PRIMARY INTERFACE (used by baselines)
    # ─────────────────────────────────────────

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
        Generate text from a prompt.

        Args:
            prompt:     User message / instruction
            system:     Optional system prompt override
            task_id:    For token tracking (e.g. "BigCodeBench/42")
            method:     Method name for tracking (e.g. "self_planning")
            call_type:  Stage label: "generation" | "planning" | "explanation" | "repair"
            attempt:    Retry index within a task (0 = first attempt)

        Returns:
            Generated text string (no post-processing)
        """
        result = self._call_api(prompt, system or SYSTEM_DEFAULT)

        if self.tracker and task_id:
            self.tracker.record(
                task_id=task_id,
                method=method,
                call_type=call_type,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                attempt=attempt,
            )

        return result.text

    # ─────────────────────────────────────────
    # STATEGEN COMPATIBILITY
    # These keep StateController working unchanged.
    # ─────────────────────────────────────────

    def generate_for_state(
        self,
        state: State,
        context: ProgramContext,
        error: Optional[str] = None,
        *,
        task_id: str = "",
        method: str = "stategen",
        attempt: int = 0,
    ) -> str:
        prompt = self._build_state_prompt(state, context, error)
        return self.generate(
            prompt,
            task_id=task_id,
            method=method,
            call_type="generation",
            attempt=attempt,
        )

    def generate_decomposition(
        self,
        problem: str,
        *,
        task_id: str = "",
        method: str = "stategen",
    ) -> str:
        prompt = f"""You are an expert Python programmer. Decompose the following programming problem into sequential states.
Each state must have:
- A short description
- Preconditions (what must already exist)
- Postconditions (what this state produces)

Problem:
{problem}

Output each state exactly like this (repeat for each state):
STATE <n>
Description: <one line>
Preconditions: <comma-separated list, or "none">
Postconditions: <comma-separated list>
---
"""
        return self.generate(
            prompt,
            task_id=task_id,
            method=method,
            call_type="decomposition",
        )

    # ─────────────────────────────────────────
    # API DISPATCH
    # ─────────────────────────────────────────

    def _call_api(self, prompt: str, system: str) -> _GenerationResult:
        if self.provider in ("openai", "openai_compatible", "azure_openai"):
            return self._call_openai(prompt, system)
        else:
            return self._call_anthropic(prompt, system)

    def _call_openai(self, prompt: str, system: str) -> _GenerationResult:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _GenerationResult(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def _call_anthropic(self, prompt: str, system: str) -> _GenerationResult:
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return _GenerationResult(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # ─────────────────────────────────────────
    # STATE PROMPT BUILDER (for StateGen)
    # ─────────────────────────────────────────

    def _build_state_prompt(
        self,
        state: State,
        context: ProgramContext,
        error: Optional[str],
    ) -> str:
        parts = []

        if not context.is_empty():
            parts.append("# Code written so far (DO NOT rewrite this):")
            parts.append(context.full_code())
            parts.append("")

        if context.variables:
            scope = ", ".join(f"{k} ({v})" for k, v in context.variables.items())
            parts.append(f"# Variables in scope: {scope}")
            parts.append("")

        parts.append(f"# Task (State {state.index}): {state.description}")
        if state.preconditions:
            parts.append(f"# Preconditions : {', '.join(state.preconditions)}")
        if state.postconditions:
            parts.append(f"# Postconditions: {', '.join(state.postconditions)}")
        parts.append("")

        if error:
            parts.append("# Previous attempt FAILED with this error:")
            parts.append(f"# {error}")
            parts.append("# Fix the error and write correct code.")
            parts.append("")

        parts.append("# Write ONLY the Python code for this task. No explanation.")
        return "\n".join(parts)
