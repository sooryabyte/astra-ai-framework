from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
import os

ProviderName = Literal["ollama", "openai", "gemini", "anthropic"]

@dataclass
class ModelConfig:
    provider: ProviderName = "ollama"
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 4096
    stream: bool = True
    # provider-specific overrides
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AstraSettings:
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))