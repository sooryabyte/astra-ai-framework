from __future__ import annotations
from typing import List
from .agent import Agent
from .tasks import Task
from .tools import Tool
from .llms import BaseLLM

class Application:
    """Coordinates agents, tasks, tools, and an LLM backend."""

    def __init__(self, agents: List[Agent], tasks: List[Task], tools: List[Tool], llm: BaseLLM):
        self.agents = agents
        self.tasks = tasks
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm

    def run(self):
        """Simple sequential execution loop for all tasks."""
        results = {}
        for task in self.tasks:
            agent = task.agent
            print(f"\n[Task] {task.description} (Agent: {agent.name})")
            result = agent.execute(task, self.llm, self.tools)
            results[task.description] = result
            print(f"[Result] {result}\n")
        return results