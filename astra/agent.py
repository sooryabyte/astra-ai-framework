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
        self.llm = llm or OllamaLLM()
        self.tools = tools or []
        self.goal = goal or ""

    def act(self, task: str) -> str:
        context = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task}"
        return self.llm.generate(context)

    def run(self, prompt: str, llm: Optional[BaseLLM] = None) -> str:
        backend = llm or self.llm
        context = f"Role: {self.role}\nGoal: {self.goal}\nPrompt: {prompt}"
        return backend.generate(context)

    def execute(self, task, llm: Optional[BaseLLM] = None, tools: Optional[dict] = None, context: Optional[str] = None) -> str:
        prompt = task.description
        if getattr(task, "expected_output", None):
            prompt += f"\n\nExpected output: {task.expected_output}"
        # prepend context (previous agents' outputs) when present
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"
        # allow per-task tool injection
        if tools:
            # For now, expose tool names in the prompt for the LLM to use (simple integration)
            prompt += f"\n\nAvailable tools: {', '.join(tools.keys())}"
        return self.run(prompt, llm=llm)
