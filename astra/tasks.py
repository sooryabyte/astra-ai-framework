from __future__ import annotations

class Task:
    def __init__(self, description: str, agent):
        self.description = description
        self.agent = agent