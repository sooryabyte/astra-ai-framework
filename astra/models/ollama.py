from __future__ import annotations
from typing import AsyncIterator, List
import httpx, json
from .base import LLMProvider
from ..messages import Message

class OllamaProvider(LLMProvider):
    async def complete(self, messages: List[Message]) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.settings.ollama_host}/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                    "stream": False,
                    **self.cfg.extra,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")

    async def stream(self, messages: List[Message]):
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.settings.ollama_host}/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                    "stream": True,
                    **self.cfg.extra,
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield obj.get("message", {}).get("content", "")
                    except Exception:
                        continue