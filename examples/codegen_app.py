from astra.agent import Agent
from astra.application import Application
from astra.llms import OllamaLLM
from astra.task import Task
from astra.tools import WriteFileTool, PythonREPLTool, ShellTool

# ----------------------------------------------------------------------------
# Define Agents
# ----------------------------------------------------------------------------
developer = Agent(
    name="Developer",
    role="Writes Python code for requested functionality",
    goal="Generate correct and efficient code.",
)

executor = Agent(
    name="Executor",
    role="Executes code and reports results",
    goal="Run code safely and return outputs or errors.",
)

# ----------------------------------------------------------------------------
# Define Tasks
# ----------------------------------------------------------------------------
task1 = Task(
    description="Generate a Python function that calculates Fibonacci numbers.",
    expected_output="A Python function definition.",
    agent=developer,
)

task2 = Task(
    description="Test the generated Fibonacci function by running it.",
    expected_output="Execution result of Fibonacci function.",
    agent=executor,
)

# ----------------------------------------------------------------------------
# LLM Backend (use local Ollama model by default)
# ----------------------------------------------------------------------------
# Use the local Ollama model you pulled (e.g. qwen2.5-coder:7b)
llm = OllamaLLM(model="qwen2.5-coder:7b")

# ----------------------------------------------------------------------------
# Application setup
# ----------------------------------------------------------------------------
app = Application(
    agents=[developer, executor],
    tasks=[task1, task2],
    tools=[WriteFileTool, PythonREPLTool, ShellTool],  # ✅ pass tools directly
    llm=llm,
)

# ----------------------------------------------------------------------------
# Run application
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    results = app.run()
    print("\n=== FINAL RESULTS ===")
    for r in results:
        print(r)