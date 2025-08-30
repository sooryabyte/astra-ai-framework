"""Tooling system for Astra agents.

Provides a @tool decorator and built-in tools (Python REPL, Shell).
All tools are strictly typed with Pydantic schemas for reliability.
"""
from __future__ import annotations
import subprocess, sys, io, contextlib
from typing import Callable, Any, Dict, Optional
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# Tool class
# ----------------------------------------------------------------------------
class Tool:
    def __init__(self, name: str, func: Callable[[BaseModel], Any], args_model: type[BaseModel], description: str):
        self.name = name
        self.func = func
        self.args_model = args_model
        self.description = description

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }

    def __call__(self, raw_args: Dict[str, Any]) -> Any:
        args = self.args_model(**raw_args)
        return self.func(args)

# ----------------------------------------------------------------------------
# Decorator
# ----------------------------------------------------------------------------
def tool(args_model: type[BaseModel], description: str):
    """Register a Pydantic-typed function as a Tool.

    Example:
        class WriteArgs(BaseModel):
            path: str
            content: str

        @tool(WriteArgs, description="Write file")
        def write_file(args: WriteArgs):
            ...
    """
    def decorator(func: Callable[[BaseModel], Any]):
        return Tool(func.__name__, func, args_model, description)
    return decorator

# ----------------------------------------------------------------------------
# Built-in tools
# ----------------------------------------------------------------------------
class PythonREPLArgs(BaseModel):
    code: str

@tool(PythonREPLArgs, description="Execute Python code in a sandboxed REPL.")
def PythonREPLTool(args: PythonREPLArgs) -> str:
    buf = io.StringIO()
    local_vars: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(args.code, {}, local_vars)
    except Exception as e:
        return f"Error: {e}\n{buf.getvalue()}"
    return buf.getvalue() or str(local_vars)


class ShellArgs(BaseModel):
    command: str

@tool(ShellArgs, description="Execute a shell command and return output.")
def ShellTool(args: ShellArgs) -> str:
    try:
        result = subprocess.run(args.command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Shell error ({e.returncode}): {e.stderr.strip()}"


class WriteFileArgs(BaseModel):
    path: str
    content: str

@tool(WriteFileArgs, description="Write text content to a file on disk.")
def WriteFileTool(args: WriteFileArgs) -> str:
    try:
        with open(args.path, "w", encoding="utf-8") as f:
            f.write(args.content)
        return f"✅ File written: {args.path}"
    except Exception as e:
        return f"❌ Error writing file: {e}"