"""Optional LLM layer. Every agent works without it (heuristics); with it, agents
refine prerequisites and difficulty. Swap in any provider by implementing complete_json."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol


class LLM(Protocol):
    def complete_json(self, system: str, prompt: str) -> dict[str, Any] | None: ...


class NoLLM:
    """Offline mode: agents fall back to deterministic heuristics."""
    def complete_json(self, system: str, prompt: str) -> None:
        return None


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1500):
        import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete_json(self, system: str, prompt: str) -> dict[str, Any] | None:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system + "\nRespond with a single JSON object only, no prose, no code fences.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None  # agents treat unparseable output as "no opinion"
