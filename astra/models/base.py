from __future__ import annotations
from typing import AsyncIterator, List, Dict, Any
from abc import ABC, abstractmethod
from ..messages import Message
from ..config import ModelConfig, AstraSettings

class LLMProvider(ABC):
    def __init__(self, cfg: ModelConfig, settings: AstraSettings):
        self.cfg = cfg
        self.settings = settings

    @abstractmethod
    async def complete(self, messages: List[Message]) -> str:
        ...

    @abstractmethod
    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        ...

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]