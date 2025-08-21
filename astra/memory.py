from __future__ import annotations
from typing import List
from .messages import Message

class ShortTermMemory:
    """Simple token-agnostic rolling buffer memory."""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._buf: List[Message] = []

    def add(self, msg: Message) -> None:
        self._buf.append(msg)
        if len(self._buf) > self.capacity:
            self._buf = self._buf[-self.capacity:]

    def clear(self) -> None:
        self._buf.clear()

    def dump(self) -> List[Message]:
        return list(self._buf)
