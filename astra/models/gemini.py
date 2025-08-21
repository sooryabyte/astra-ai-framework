from __future__ import annotations
from typing import AsyncIterator, List
try:
    import google.generativeai as genai
except Exception:
    genai = None
from .base import LLMProvider
from ..messages import Message

class GeminiProvider(LLMProvider):
    async def complete(self, messages: List[Message]) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.cfg.model)
        # Gemini uses a different format; join history plainly
        prompt = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
        resp = await model.generate_content_async(prompt)
        return resp.text or ""

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.cfg.model)
        prompt = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
        stream = await model.generate_content_async(prompt, stream=True)
        async for evt in stream:
            if getattr(evt, "text", None):
                yield evt.text