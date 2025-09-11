from typing import Optional
import subprocess
import asyncio

from .config import ModelConfig, AstraSettings
from .models import get_provider
from .messages import Message, Role


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# ---------------------------
# Provider-backed sync adapter
# ---------------------------
class ProviderSyncAdapter(BaseLLM):
    """Adapter that wraps the async LLMProvider implementations and exposes a sync generate()."""

    def __init__(self, cfg: Optional[ModelConfig] = None, settings: Optional[AstraSettings] = None):
        self.cfg = cfg or ModelConfig()
        self.settings = settings or AstraSettings()
        self.provider = get_provider(self.cfg, self.settings)

    def generate(self, prompt: str) -> str:
        # Build a minimal Message-like structure expected by providers

        messages = [Message(role=Role.USER, content=prompt)]

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.provider.complete(messages))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            asyncio.set_event_loop(None)


# ---------------------------
# Backwards-compatible specific wrappers
# ---------------------------
class OllamaLLM(BaseLLM):
    """Local Ollama LLM backend (keeps original behavior)."""

    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            return result.stdout.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            return f"Ollama error: {e.stderr.decode('utf-8').strip()}"



class GeminiChat(BaseLLM):
    """Wrapper for Google Gemini models; prefers provider-based implementation."""

    def __init__(self, model: str = "models/gemini-2.5-pro", api_key: Optional[str] = None):
        cfg = ModelConfig(provider="gemini", model=model)
        settings = AstraSettings(gemini_api_key=api_key)
        self.adapter = ProviderSyncAdapter(cfg, settings)

    def generate(self, prompt: str) -> str:
        return self.adapter.generate(prompt)


class OpenAIChat(BaseLLM):
    """Wrapper for OpenAI Chat models using provider abstraction (non-streaming)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        cfg = ModelConfig(provider="openai", model=model)
        settings = AstraSettings(openai_api_key=api_key)
        self.adapter = ProviderSyncAdapter(cfg, settings)

    def generate(self, prompt: str) -> str:
        return self.adapter.generate(prompt)


class AnthropicChat(BaseLLM):
    """Wrapper for Anthropic Claude models through provider abstraction."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: Optional[str] = None):
        cfg = ModelConfig(provider="anthropic", model=model)
        settings = AstraSettings(anthropic_api_key=api_key)
        self.adapter = ProviderSyncAdapter(cfg, settings)

    def generate(self, prompt: str) -> str:
        return self.adapter.generate(prompt)
