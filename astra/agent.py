from typing import List, Optional, Dict, Any
import json
import re
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

    def run(self, prompt: str, llm: Optional[BaseLLM] = None) -> str:
        backend = llm or self.llm
        context = f"Role: {self.role}\nGoal: {self.goal}\nPrompt: {prompt}"
        return backend.generate(context)

    def execute(self, task, llm: Optional[BaseLLM] = None, tools: Optional[Dict[str, Tool]] = None, context: Optional[str] = None) -> str:
        backend = llm or self.llm

        system_instructions = (
            "You are an agent that may use tools.\n"
            "To call a tool, reply with ONLY a single JSON object:\n"
            "{\"tool\": \"<ToolName>\", \"args\": { ... }}\n"
            "When you are completely done, reply with: FINAL: <your final answer> (on its own line).\n"
            "Do not add any extra text outside JSON or the FINAL line.\n\n"
            "Tool usage guidance (language-agnostic):\n"
            "- Always use ExtractCodeBlockTool to pull the latest fenced code from previous outputs.\n"
            "- Read 'normalized_language' from the extractor result and pass that to PistonExecuteTool. If it's null, fall back to the raw 'language' or infer from code heuristics.\n"
            "- Use PistonExecuteTool to run code in ANY supported language (python, c, cpp, java, go, rust, etc.).\n"
            "- Provide stdin explicitly for programs that read input; separate lines with \n.\n"
            "- Do NOT spawn subprocesses inside the code under test. Run the user program directly with Piston per test case.\n"
            "- PythonREPLTool is only for quick internal calculations or prototypes, not for validating the user program.\n\n"
            "Examples:\n"
            "1) Extract code then execute with detected language and inputs:\n"
            "{\n  \"tool\": \"ExtractCodeBlockTool\",\n  \"args\": {\n    \"text\": \"...previous output with code...\"\n  }\n}\n"
            "Then (pseudo, replace <lang> with normalized_language):\n"
            "{\n  \"tool\": \"PistonExecuteTool\",\n  \"args\": {\n    \"language\": \"<lang>\",\n    \"code\": \"<extracted code>\",\n    \"stdin\": \"case input here\\n\"\n  }\n}\n"
            "2) C++ example via Piston:\n"
            "{\n  \"tool\": \"PistonExecuteTool\",\n  \"args\": {\n    \"language\": \"cpp\",\n    \"code\": \"#include <bits/stdc++.h>\\nusing namespace std; int main(){int a,b; if(!(cin>>a>>b)) return 0; cout<<a+b; }\",\n    \"stdin\": \"2 3\\n\"\n  }\n}\n"
        )

        # Build initial user prompt
        user_prompt = task.description
        if getattr(task, "expected_output", None):
            user_prompt += f"\n\nExpected output: {task.expected_output}"
        if context:
            user_prompt = f"Context from previous steps:\n{context}\n\nTask:\n{user_prompt}"

        # If tools available, show their schemas to help the model call them properly
        tool_block = ""
        if tools:
            try:
                schemas = [t.schema() for t in tools.values()]
                tool_block = "\n\nAvailable tools (name and JSON schema):\n" + json.dumps(schemas, ensure_ascii=False)
            except Exception:
                tool_block = "\n\nAvailable tools: " + ", ".join(tools.keys())

        conversation = f"Role: {self.role}\nGoal: {self.goal}\n\n{system_instructions}\n\n{user_prompt}{tool_block}"

        # Helper to parse a potential tool call JSON from the model output
        def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
            s = (text or "").strip()
            # detect FINAL anywhere in the string
            idx = s.find("FINAL:")
            if idx != -1:
                return {"final": s[idx + len("FINAL:"):].strip()}

            # 1) fenced ```json ... ``` block
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except Exception:
                    pass

            # 2) try strict parse of entire string (rare but cheap)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                    return obj
            except Exception:
                pass

            # 3) find minimal JSON-looking objects and test each
            for m in re.finditer(r"\{[\s\S]*?\}", s):
                seg = m.group(0)
                if '"tool"' not in seg or '"args"' not in seg:
                    continue
                try:
                    obj = json.loads(seg)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except Exception:
                    continue
            return None

        # Tool-calling loop with a small cap to avoid infinite cycles
        max_steps = 6
        transcript = conversation
        last_final: Optional[str] = None
        last_tool_result: Optional[str] = None

        for _ in range(max_steps):
            reply = backend.generate(transcript)
            parsed = parse_tool_call(reply)
            if not parsed:
                # treat as final if model didn't follow protocol
                last_final = reply.strip()
                break
            if "final" in parsed:
                last_final = str(parsed["final"]).strip()
                break
            # Execute tool call
            tool_name = str(parsed.get("tool", ""))
            if not tools or tool_name not in tools:
                last_final = f"Tool '{tool_name}' not available. Proceeding without tools.\n\n{reply}"
                break
            try:
                tool_obj = tools[tool_name]
                tool_args = parsed.get("args", {})
                tool_result = tool_obj(tool_args)
                last_tool_result = str(tool_result)
            except Exception as e:
                tool_result = f"Tool execution error: {e}"
                last_tool_result = str(tool_result)

            # Feed tool result back to the model
            transcript += (
                f"\n\n[Tool call]\nName: {tool_name}\nArgs: {json.dumps(tool_args, ensure_ascii=False)}\n"
                f"[Tool result]\n{tool_result}\n\n"
                "Respond with another tool call JSON or FINAL: <answer>."
            )

        return (last_final or last_tool_result or "").strip()
