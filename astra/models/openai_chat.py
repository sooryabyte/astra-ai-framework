from __future__ import annotations
from typing import AsyncIterator, List
import os
try:
    import openai  # official SDK v1
except Exception:  # optional
    openai = None
from .base import LLMProvider
from ..messages import Message

class OpenAIChatProvider(LLMProvider):
    async def complete(self, messages: List[Message]) -> str:
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai>=1.40.0")
        client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai>=1.40.0")
        client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        stream = await client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta