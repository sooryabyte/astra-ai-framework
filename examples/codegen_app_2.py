from astra.agent import Agent
from astra.task import Task
from astra.tools import PythonREPLTool
from astra.llms import OllamaLLM

# Initialize LLM (use the model you pulled)
llm = OllamaLLM(model="qwen2.5-coder:7b")

# Agents
developer = Agent(
    name="Developer",
    role="Writes and fixes Python code",
    goal="Expert software engineer who generates correct, efficient code.",
    llm=llm,
)

executor = Agent(
    name="Executor",
    role="Runs Python code and reports results",
    goal="Responsible for executing code safely and returning output/errors.",
    tools=[PythonREPLTool],
    llm=llm,
)

# Function for self-correcting workflow
def run_with_correction(prompt, max_retries=3):
    for attempt in range(max_retries):
        print(f"\n=== Attempt {attempt+1} ===")

        # Step 1: Developer writes code
        task_dev = Task(description=prompt, agent=developer)
        code = task_dev.run()

        print("\n[Developer Output]\n", code)

        # Step 2: Executor runs code
        task_exec = Task(
            description=f"Run the following Python code and report result:\n{code}",
            agent=executor,
        )
        # Provide the executor with the tools mapping so it can reference/use tools
        tools_map = {PythonREPLTool.name: PythonREPLTool}
        result = executor.execute(task_exec, llm=llm, tools=tools_map)

        print("\n[Executor Output]\n", result)

        # If error, retry with feedback
        if "Traceback" in result or "Error" in result:
            print("\n⚠️ Execution failed, sending error back to Developer...\n")
            prompt = f"Fix the following Python code based on the error:\n\nCode:\n{code}\n\nError:\n{result}"
        else:
            print("\n✅ Success!")
            break


if __name__ == "__main__":
    run_with_correction("Generate a Python function that calculates Fibonacci numbers.")
