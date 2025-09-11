from __future__ import annotations
from typing import AsyncIterator, List
import asyncio
try:
    import google.generativeai as genai
    from google.api_core import exceptions as ga_exceptions
except Exception:
    genai = None
    ga_exceptions = None
from .base import LLMProvider
from ..messages import Message


def _join_messages(messages: List[Message]) -> str:
    return "\n".join(f"{m.role.value}: {m.content}" for m in messages)


class GeminiProvider(LLMProvider):
    async def complete(self, messages: List[Message]) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=self.settings.gemini_api_key)

        prompt = _join_messages(messages)
        # Trim very long prompts to avoid backend errors (keep tail context)
        MAX_CHARS = 20000
        if len(prompt) > MAX_CHARS:
            prompt = prompt[-MAX_CHARS:]

        # Primary model
        model = genai.GenerativeModel(self.cfg.model)

        # Retry with exponential backoff on transient server errors
        max_retries = 3
        delay = 1.0
        for attempt in range(max_retries):
            try:
                # enforce a client-side timeout to avoid indefinite waits
                resp = await asyncio.wait_for(model.generate_content_async(prompt), timeout=60)
                return resp.text or ""
            except Exception as e:
                # Immediately surface NotFound (bad model) errors
                if ga_exceptions is not None and isinstance(e, ga_exceptions.NotFound):
                    raise

                # If transient server-side error, retry
                if ga_exceptions is not None and isinstance(e, (ga_exceptions.InternalServerError, ga_exceptions.ServiceUnavailable, ga_exceptions.DeadlineExceeded)):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    # last attempt failed; fall back to cheaper/flash models where possible
                    fallbacks = [
                        "models/gemini-2.5-flash",
                        "models/gemini-2.5-flash-lite",
                        "models/gemini-1.5-flash",
                    ]
                    for fb in fallbacks:
                        try:
                            fb_model = genai.GenerativeModel(fb)
                            resp = await asyncio.wait_for(fb_model.generate_content_async(prompt), timeout=60)
                            return resp.text or ""
                        except Exception:
                            continue
                    # exhausted fallbacks, re-raise last exception
                    raise
                # unknown exception: re-raise
                raise

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=self.settings.gemini_api_key)

        prompt = _join_messages(messages)
        MAX_CHARS = 20000
        if len(prompt) > MAX_CHARS:
            prompt = prompt[-MAX_CHARS:]

        model = genai.GenerativeModel(self.cfg.model)
        # streaming may also encounter transient errors; keep a single attempt for stream
        try:
            stream = await model.generate_content_async(prompt, stream=True)
            async for evt in stream:
                if getattr(evt, "text", None):
                    yield evt.text
        except Exception as e:
            # surface NotFound immediately
            if ga_exceptions is not None and isinstance(e, ga_exceptions.NotFound):
                raise
            # otherwise convert to a runtime error for caller
            raise RuntimeError(f"Gemini streaming failed: {e}")