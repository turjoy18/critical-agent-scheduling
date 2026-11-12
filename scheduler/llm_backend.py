"""Free-tier LLM backends behind a common interface (#14).

Cost policy: chat/completions must use free tiers only. Paid budget is GPU-only.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage
    model: str
    provider: str


class LLMBackend(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


def _chat(
    client: object,
    model: str,
    provider: str,
    system: str,
    user: str,
    max_retries: int = 5,
) -> LLMResponse:
    import openai
    from openai import OpenAI

    assert isinstance(client, OpenAI)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            usage = LLMUsage(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            )
            return LLMResponse(content, usage, model, provider)
        except openai.RateLimitError:
            if attempt >= max_retries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError("unreachable")


class GroqBackend:
    provider = "groq"

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set (free-tier Groq)")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

    def complete(self, system: str, user: str) -> LLMResponse:
        return _chat(self._client, self.model, self.provider, system, user)


class GeminiOpenAICompatBackend:
    """Google AI Studio via OpenAI-compatible endpoint when available."""

    provider = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (free-tier Google AI Studio)")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=key, base_url=base_url)

    def complete(self, system: str, user: str) -> LLMResponse:
        return _chat(self._client, self.model, self.provider, system, user)


class OpenRouterFreeBackend:
    provider = "openrouter"

    def __init__(
        self,
        model: str = "google/gemma-2-9b-it:free",
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    def complete(self, system: str, user: str) -> LLMResponse:
        return _chat(self._client, self.model, self.provider, system, user)


def get_backend(name: str | None = None) -> LLMBackend:
    """Resolve free-tier backend. Order: explicit name → env LLM_BACKEND → groq → gemini → openrouter."""
    choice = (name or os.getenv("LLM_BACKEND") or "").strip().lower()
    if choice and choice not in ("groq", "gemini", "openrouter"):
        raise ValueError(f"Unknown LLM_BACKEND '{choice}'")
    providers = [choice] if choice else ["groq", "gemini", "openrouter"]
    factories = {
        "groq": GroqBackend,
        "gemini": GeminiOpenAICompatBackend,
        "openrouter": OpenRouterFreeBackend,
    }
    errors: list[str] = []
    for p in providers:
        try:
            return factories[p]()  # type: ignore[return-value]
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError(
        "No free-tier LLM backend configured. Set GROQ_API_KEY, GEMINI_API_KEY, "
        f"or OPENROUTER_API_KEY. Details: {errors}"
    )
