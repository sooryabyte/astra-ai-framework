from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

class Task:
    def __init__(
        self,
        description: str,
        agent: "Agent",
        expected_output: Optional[str] = None,
        tools: Optional[List] = None,
    ):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.tools = tools or []

    # Tasks are executed through Application -> Agent.execute().