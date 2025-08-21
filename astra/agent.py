# === path: astra/agent.py ===
from typing import List, Optional
from .tools import Tool
from .llms import BaseLLM, OllamaLLM

class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        llm: Optional[BaseLLM] = None,
        tools: Optional[List[Tool]] = None,
        goal: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.llm = llm or OllamaLLM()  # ✅ default is always Ollama
        self.tools = tools or []
        self.goal = goal or ""

    def act(self, task: str) -> str:
        context = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task}"
        return self.llm.generate(context)

    def run(self, prompt: str, llm: Optional[BaseLLM] = None) -> str:
        """Generate a response for the given prompt using the provided LLM or the agent's LLM."""
        backend = llm or self.llm
        context = f"Role: {self.role}\nGoal: {self.goal}\nPrompt: {prompt}"
        return backend.generate(context)

    def execute(self, task, llm: Optional[BaseLLM] = None, tools: Optional[dict] = None) -> str:
        """High-level entry used by Application: accepts a Task instance and returns a string result."""
        prompt = task.description
        if getattr(task, "expected_output", None):
            prompt += f"\n\nExpected output: {task.expected_output}"
        # allow per-task tool injection
        if tools:
            # For now, expose tool names in the prompt for the LLM to use (simple integration)
            prompt += f"\n\nAvailable tools: {', '.join(tools.keys())}"
        return self.run(prompt, llm=llm)
