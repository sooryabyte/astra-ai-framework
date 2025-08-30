from typing import Optional, List

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

    def run(self) -> str:
        """Execute the task using its assigned agent."""
        prompt = self.description
        if self.expected_output:
            prompt += f"\n\nExpected output: {self.expected_output}"
        return self.agent.run(prompt)