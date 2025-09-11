from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
from .messages import Message

@dataclass
class HandOff:
    to_agent: str
    reason: str

class Router:
    def __init__(self, decide: Callable[[Message], Optional[HandOff]]):
        self.decide = decide

    def route(self, msg: Message) -> Optional[HandOff]:
        return self.decide(msg)