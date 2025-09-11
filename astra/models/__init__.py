from .base import LLMProvider
from .ollama import OllamaProvider
from .openai_chat import OpenAIChatProvider
from .gemini import GeminiProvider
from .anthropic import AnthropicProvider
from ..config import ModelConfig, AstraSettings

def get_provider(cfg: ModelConfig, settings: AstraSettings) -> LLMProvider:
    if cfg.provider == "ollama":
        return OllamaProvider(cfg, settings)
    if cfg.provider == "openai":
        return OpenAIChatProvider(cfg, settings)
    if cfg.provider == "gemini":
        return GeminiProvider(cfg, settings)
    if cfg.provider == "anthropic":
        return AnthropicProvider(cfg, settings)
    raise ValueError(f"Unknown provider: {cfg.provider}")
