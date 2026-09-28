"""LLM interface plus one concrete provider client.

Provider-specific code lives only in OpenAIClient; the rest of the project
depends on the LLMClient interface.
"""

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()  # optional .env support; OPENAI_API_KEY may also come from the shell


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIClient:
    """Reads OPENAI_API_KEY from the environment; model is overridable via OPENAI_MODEL."""

    def __init__(self, model=None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
