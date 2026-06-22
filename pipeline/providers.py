"""
Pluggable LLM provider interface.

All pipeline stages (datagen, VJ filter, cluster description, patch gen) go
through this interface. Never instantiate a vendor client inside pipeline code
— always inject a provider.

Swap in a local model to keep all training data on-premises:
  base_url="http://localhost:11434/v1", api_key="ollama", model="qwen2.5:7b"

Environment variables (read by provider_from_env):
  TOOLHIVE_PROVIDER_API_KEY   (required; use "none" for local models)
  TOOLHIVE_PROVIDER_MODEL     (required)
  TOOLHIVE_PROVIDER_BASE_URL  (required; OpenAI-compatible endpoint)

Examples:
  Groq:   base_url=https://api.groq.com/openai/v1
  Gemini: base_url=https://generativelanguage.googleapis.com/v1beta/openai/
  Ollama: base_url=http://localhost:11434/v1
  vLLM:   base_url=http://localhost:8000/v1
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    base_url: str


class LLMProvider:
    """
    Thin wrapper around an OpenAI-compatible chat completions endpoint.
    Injected into every pipeline function — never instantiated inside them.
    """

    def __init__(self, config: ProviderConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'toolhive[pipeline]'"
            ) from e
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: str | None = None,
    ) -> str:
        """
        Send messages and return the assistant's text reply.
        Retries twice on rate-limit errors with exponential backoff.
        All other errors propagate immediately.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt, wait in enumerate([0, 0.5, 1.0]):
            try:
                if wait:
                    time.sleep(wait)
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                # Only retry on rate limit errors
                if "rate" in str(e).lower() and attempt < 2:
                    last_err = e
                    continue
                raise
        raise last_err  # type: ignore[misc]


def provider_from_env() -> LLMProvider:
    """
    Build a provider from environment variables.
    Raises ValueError with a clear message if any required var is missing.
    """
    missing = [
        v for v in ("TOOLHIVE_PROVIDER_API_KEY", "TOOLHIVE_PROVIDER_MODEL", "TOOLHIVE_PROVIDER_BASE_URL")
        if not os.environ.get(v)
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            "Set TOOLHIVE_PROVIDER_API_KEY, TOOLHIVE_PROVIDER_MODEL, and "
            "TOOLHIVE_PROVIDER_BASE_URL before running the pipeline."
        )
    return LLMProvider(
        ProviderConfig(
            api_key=os.environ["TOOLHIVE_PROVIDER_API_KEY"],
            model=os.environ["TOOLHIVE_PROVIDER_MODEL"],
            base_url=os.environ["TOOLHIVE_PROVIDER_BASE_URL"],
        )
    )
