import streamlit as st
import os
from astra.agent import Agent
from astra.application import Application
from astra.llms import GeminiChat, OllamaLLM, OpenAIChat, AnthropicChat
from astra.task import Task
from astra.tools import PythonREPLTool, WriteFileTool, PistonExecuteTool, ExtractCodeBlockTool, ExtractTestCasesTool

st.set_page_config(page_title="CodeSmith - Multi-Agent Codegen", layout="wide")

st.title("CodeSmith - AI Code Generator")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model Backend",
    ["Ollama (local)", "Gemini (cloud)", "OpenAI (cloud)", "Anthropic (cloud)"]
)

api_key = None
if model_choice == "Gemini (cloud)":
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
elif model_choice == "OpenAI (cloud)":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elif model_choice == "Anthropic (cloud)":
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")

# --- User Prompt ---
prompt = st.text_area(
    "💡 Enter your code generation request:",
    placeholder="e.g. Generate a Python function that calculates factorial..."
)

run_button = st.button("Run Application")

# --- Define Agents ---
developer = Agent(
    name="Developer",
    role="Writes code for requested functionality in any language",
    goal="Generate correct and efficient code.",
)

executor = Agent(
    name="Executor",
    role="Executes code and reports results",
    goal="Run code safely and return outputs or errors.",
)

explainer = Agent(
    name="Explainer",
    role="Explains code and its functionality.",
    goal="Explain the code for better understanding of user.",
)

if run_button and prompt:
    with st.spinner("Running agents..."):

        # --- Pick LLM backend ---
        if model_choice == "Ollama (local)":
            llm = OllamaLLM(model="qwen2.5-coder:7b")
        elif model_choice == "Gemini (cloud)":
            if not api_key:
                st.error("Please enter your Gemini API Key.")
                st.stop()
            llm = GeminiChat(
                model=os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash"),
                api_key=api_key,
            )
        elif model_choice == "OpenAI (cloud)":
            if not api_key:
                st.error("Please enter your OpenAI API Key.")
                st.stop()
            llm = OpenAIChat(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=api_key,
            )
        else:  # Anthropic
            if not api_key:
                st.error("Please enter your Anthropic API Key.")
                st.stop()
            llm = AnthropicChat(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                api_key=api_key,
            )

        # --- Define tasks dynamically from user prompt ---
        task1 = Task(
            description=prompt,
            expected_output="A complete and working code snippet.",
            agent=developer,
        )

        task2 = Task(
            description=(
        "Extract the latest fenced code block and its language, then test the generated code by running it via Piston using stdin examples. "
        "For interactive programs (input/choice loops), provide appropriate inputs and include an exit."
            ),
            expected_output="Execution result including stdout and any errors.",
            agent=executor,
        )

        task3 = Task(
            description="Explain the working of the finished final code output.",
            expected_output="Explanation of generated code.",
            agent=explainer,
        )

        # --- Build Application ---
        app = Application(
            agents=[developer, executor, explainer],
            tasks=[task1, task2, task3],
            tools=[PythonREPLTool, WriteFileTool, PistonExecuteTool, ExtractCodeBlockTool, ExtractTestCasesTool],
            llm=llm,
        )

        results = app.run()

        st.success("✅ Finished Running!")

        st.markdown("## Results by agent")
        sections = [
            ("🧑‍💻 Developer", developer, task1),
            ("🧪 Executor", executor, task2),
            ("📘 Explainer", explainer, task3),
        ]
        for title, agent_obj, task_obj in sections:
            with st.container(border=True):
                st.subheader(f"{title}")
                st.caption(f"Role: {agent_obj.role}")
                out = results.get(task_obj.description, "")
                if not out:
                    st.info("No output.")
                else:
                    st.markdown(out)

        # Optional raw debug
        with st.expander("🔍 Show raw results (debug)"):
            try:
                st.write(results)
            except Exception:
                st.text(str(results))
