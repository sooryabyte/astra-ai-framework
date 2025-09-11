from __future__ import annotations
from typing import AsyncIterator, List
import asyncio
try:
    from anthropic import AsyncAnthropic, APIStatusError
except Exception:  # package optional until user selects provider
    AsyncAnthropic = None  # type: ignore
    APIStatusError = Exception  # type: ignore

from .base import LLMProvider
from ..messages import Message


def _join_messages(messages: List[Message]) -> List[dict]:
    # Anthropic expects a system prompt separately and user/assistant turns; we collapse to one user turn.
    # Simplicity: merge all into one user content block.
    content = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
    return [{"role": "user", "content": content}]


class AnthropicProvider(LLMProvider):
    async def complete(self, messages: List[Message]) -> str:
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed. pip install anthropic>=0.30.0")
        client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": _join_messages(messages),
        }
        # Basic retry for transient 5xx
        attempt, delay, max_attempts = 0, 1.0, 3
        last_err = None
        while attempt < max_attempts:
            try:
                resp = await asyncio.wait_for(client.messages.create(**body), timeout=60)
                # Anthropic returns a list of content blocks
                if resp.content and len(resp.content) > 0:
                    # concatenate text blocks
                    return "".join(part.text for part in resp.content if getattr(part, 'text', None))
                return ""
            except Exception as e:  # refine if anthropic exposes status
                last_err = e
                if isinstance(e, APIStatusError) and getattr(e, 'status_code', 500) >= 500 and attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    attempt += 1
                    continue
                raise
        if last_err:
            raise last_err
        return ""

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed. pip install anthropic>=0.30.0")
        client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": _join_messages(messages),
            "stream": True,
        }
        try:
            stream = await client.messages.create(**body)
            async for evt in stream:
                if hasattr(evt, 'delta') and getattr(evt.delta, 'text', None):
                    yield evt.delta.text
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming failed: {e}")
