from __future__ import annotations
from typing import List
from .agent import Agent
from .task import Task
from .tools import Tool
from .llms import BaseLLM

class Application:
    def __init__(self, agents: List[Agent], tasks: List[Task], tools: List[Tool], llm: BaseLLM):
        self.agents = agents
        self.tasks = tasks
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm

    def run(self):
        results = {}
        accumulated_context_lines: list[str] = []
        for task in self.tasks:
            agent = task.agent
            print(f"\n[Task] {task.description} (Agent: {agent.name})")
            context_blob = "\n".join(accumulated_context_lines) if accumulated_context_lines else None
            result = agent.execute(task, self.llm, self.tools, context=context_blob)
            results[task.description] = result
            accumulated_context_lines.append(f"{agent.name} result: {result}")
            print(f"[Result] {result}\n")
        return results