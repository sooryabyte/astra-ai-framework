"""Tooling system for Astra agents.

Provides a @tool decorator and built-in tools (Python REPL, Shell).
All tools are strictly typed with Pydantic schemas for reliability.
"""
from __future__ import annotations
import subprocess, io, contextlib, os
from typing import Callable, Any, Dict, Optional, List
from pydantic import BaseModel
import json
import httpx
import re

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
    cmd = args.command.strip()
    if os.name == "nt":  # Windows
        # Quick guard for POSIX-only source command
        if cmd.startswith(". "):
            return "Shell error: POSIX '. <file>' (source) is not supported on Windows PowerShell."
        # Translate common POSIX path prefix to Windows style
        if cmd.startswith("./"):
            cmd = ".\\" + cmd[2:]
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                capture_output=True,
                text=True,
                check=True,
            )
            return (result.stdout or result.stderr).strip()
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            return f"Shell error ({e.returncode}): {err}"
    else:
        # On POSIX, run via bash -lc for better compatibility
        try:
            result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=True)
            return (result.stdout or result.stderr).strip()
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            return f"Shell error ({e.returncode}): {err}"


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


# ----------------------------------------------------------------------------
# Piston Execute Tool (Public API, no Docker required)
# ----------------------------------------------------------------------------
class PistonFile(BaseModel):
    name: Optional[str] = None
    content: str
    encoding: Optional[str] = None  # 'utf8' | 'base64' | 'hex'


class PistonExecuteArgs(BaseModel):
    language: str
    code: Optional[str] = None  # convenience for single-file snippets
    version: Optional[str] = "*"  # SemVer or '*'
    files: Optional[List[PistonFile]] = None  # if provided, overrides code
    stdin: Optional[str] = None
    args: Optional[List[str]] = None
    run_timeout: Optional[int] = 3000  # ms
    compile_timeout: Optional[int] = 10000  # ms


@tool(
    PistonExecuteArgs,
    description=(
        "Run code using the public Piston API (no Docker required). "
        "Provide language (e.g. 'python', 'js', 'c', 'cpp', 'java'), optional version '*' and either 'code' for a single file or 'files' for multi-file. "
        "Optional: stdin (string), args (list of strings). Returns combined stdout/stderr and exit info."
    ),
)
def PistonExecuteTool(args: PistonExecuteArgs) -> str:
    base_url = os.environ.get("PISTON_BASE_URL", "https://emkc.org/api/v2/piston")
    url = f"{base_url}/execute"

    # Build files payload
    files_payload: List[Dict[str, str]] = []
    if args.files and len(args.files) > 0:
        for f in args.files:
            files_payload.append({
                "name": f.name or "main",
                "content": f.content,
                **({"encoding": f.encoding} if f.encoding else {}),
            })
    elif args.code:
        # Heuristic for extension name by language
        default_name = {
            "python": "main.py",
            "py": "main.py",
            "javascript": "main.js",
            "js": "main.js",
            "ts": "main.ts",
            "c": "main.c",
            "cpp": "main.cpp",
            "c++": "main.cpp",
            "java": "Main.java",
            "go": "main.go",
            "rust": "main.rs",
            "rb": "main.rb",
            "ruby": "main.rb",
            "php": "main.php",
        }.get(args.language.lower(), "main.txt")
        files_payload = [{"name": default_name, "content": args.code}]
    else:
        return "❌ PistonExecuteTool: Provide either 'code' or 'files'."

    payload = {
        "language": args.language,
        "version": args.version or "*",
        "files": files_payload,
        **({"stdin": args.stdin} if args.stdin is not None else {}),
        **({"args": args.args} if args.args is not None else {}),
        **({"run_timeout": args.run_timeout} if args.run_timeout is not None else {}),
        **({"compile_timeout": args.compile_timeout} if args.compile_timeout is not None else {}),
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
    except httpx.HTTPError as e:
        return f"❌ Piston HTTP error: {e}"
    except Exception as e:
        return f"❌ Piston error: {e}"

    # Shape: { run: { stdout, stderr, code, status, message, ... }, compile?: {...} }
    run = data.get("run", {})
    compile_ = data.get("compile", {})

    parts = []
    if compile_:
        c_out = compile_.get("stdout", "")
        c_err = compile_.get("stderr", "")
        if c_out:
            parts.append(f"[compile stdout]\n{c_out}")
        if c_err:
            parts.append(f"[compile stderr]\n{c_err}")
        status = compile_.get("status") or compile_.get("message")
        if status:
            parts.append(f"[compile status] {status}")

    r_out = run.get("stdout", "")
    r_err = run.get("stderr", "")
    if r_out:
        parts.append(f"[stdout]\n{r_out}")
    if r_err:
        parts.append(f"[stderr]\n{r_err}")

    meta = {
        "code": run.get("code"),
        "status": run.get("status"),
        "cpu_time": run.get("cpu_time"),
        "wall_time": run.get("wall_time"),
        "memory": run.get("memory"),
        "language": data.get("language"),
        "version": data.get("version"),
    }
    parts.append("[meta] " + json.dumps(meta, ensure_ascii=False))
    return "\n".join(parts).strip()


# ----------------------------------------------------------------------------
# Extract latest fenced code block helper
# ----------------------------------------------------------------------------
class ExtractCodeArgs(BaseModel):
    text: str
    prefer_language: Optional[str] = None  # e.g., 'python', 'js', 'cpp'


@tool(
    ExtractCodeArgs,
    description=(
        "Extract the most recent fenced code block from given text. "
        "Optionally prefer a language (e.g., 'python'). Returns a compact JSON string with fields: language, code."
    ),
)
def ExtractCodeBlockTool(args: ExtractCodeArgs) -> str:
    s = args.text or ""
    blocks = []
    # match ```lang\n...\n``` or ```\n...\n```; language is optional word chars/plus signs
    for m in re.finditer(r"```([\w#+-]*)\s*\n([\s\S]*?)```", s, flags=re.MULTILINE):
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2)
        blocks.append((lang, code))

    def normalize_lang(raw: Optional[str], code_text: str) -> Optional[str]:
        """Normalize various language labels to Piston names. Heuristic if missing."""
        if raw:
            r = raw.lower()
            mapping = {
                "py": "python", "python": "python",
                "js": "javascript", "javascript": "javascript", "node": "javascript", "nodejs": "javascript",
                "ts": "typescript", "typescript": "typescript",
                "c": "c",
                "cpp": "cpp", "c++": "cpp", "cc": "cpp",
                "java": "java",
                "go": "go", "golang": "go",
                "rs": "rust", "rust": "rust",
                "rb": "ruby", "ruby": "ruby",
                "php": "php",
                "cs": "csharp", "c#": "csharp", "csharp": "csharp",
                "kt": "kotlin", "kotlin": "kotlin",
                "swift": "swift",
                "sh": "bash", "bash": "bash", "shell": "bash",
                "r": "r",
                "scala": "scala",
                "dart": "dart",
                "perl": "perl",
                "haskell": "haskell", "hs": "haskell",
            }
            if r in mapping:
                return mapping[r]
        # Heuristics based on code
        ct = code_text
        if "#include <iostream>" in ct or "using namespace std" in ct:
            return "cpp"
        if "#include <stdio.h>" in ct and "printf(" in ct:
            return "c"
        if ct.startswith("#!/usr/bin/env python") or "def main(" in ct or "print(" in ct and "#include" not in ct:
            return "python"
        if "console.log(" in ct or "function(" in ct and "#include" not in ct:
            return "javascript"
        if "package main" in ct and "func main()" in ct:
            return "go"
        if "fn main()" in ct and "println!" in ct:
            return "rust"
        if "public static void main(String[] args)" in ct:
            return "java"
        return None

    chosen = None
    pref = (args.prefer_language or "").strip().lower() or None
    if blocks:
        if pref:
            for lang, code in reversed(blocks):  # prefer latest occurrence of preferred language
                if lang and (lang == pref or (lang.startswith(pref))):
                    chosen = (lang, code)
                    break
        if not chosen:
            chosen = blocks[-1]  # latest block
    else:
        # No fenced code: return empty with note
        return json.dumps({"language": None, "normalized_language": None, "code": "", "note": "no fenced code block found"})

    lang, code = chosen
    norm = normalize_lang(lang, code)
    return json.dumps({"language": lang, "normalized_language": norm, "code": code.strip()})


# ----------------------------------------------------------------------------
# Extract Test Cases helper
# ----------------------------------------------------------------------------
class ExtractTestsArgs(BaseModel):
    text: str


@tool(
    ExtractTestsArgs,
    description=(
        "Extract Product Manager style test cases from text. "
        "Looks for a JSON array of objects with fields at least 'stdin' and 'expected' (and optional 'name'). "
        "Returns a compact JSON string: {cases: [{name, stdin, expected}], count}."
    ),
)
def ExtractTestCasesTool(args: ExtractTestsArgs) -> str:
    s = args.text or ""

    def try_parse_candidates(cands):
        for cand in cands:
            try:
                data = json.loads(cand)
                if isinstance(data, list):
                    ok = []
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            ok = []
                            break
                        if "stdin" in item and "expected" in item:
                            name = item.get("name") or f"Case {i+1}"
                            ok.append({
                                "name": str(name),
                                "stdin": str(item.get("stdin", "")),
                                "expected": str(item.get("expected", "")),
                            })
                        else:
                            ok = []
                            break
                    if ok:
                        return ok
            except Exception:
                continue
        return None

    # 1) prefer fenced ```json blocks
    json_blocks = [m.group(1) for m in re.finditer(r"```json\s*\n([\s\S]*?)```", s, flags=re.IGNORECASE)]
    cases = try_parse_candidates(reversed(json_blocks)) if json_blocks else None
    if cases:
        return json.dumps({"cases": cases, "count": len(cases)})

    # 2) generic array-of-objects regex (simple, non-nested objects acceptable)
    array_objs = [m.group(1) for m in re.finditer(r"(\[\s*(?:\{[\s\S]*?\}\s*,?\s*)+\])", s)]
    cases = try_parse_candidates(reversed(array_objs)) if array_objs else None
    if cases:
        return json.dumps({"cases": cases, "count": len(cases)})

    # 3) Look near a 'Test Cases' section
    m = re.search(r"Test Cases\s*:?\s*([\s\S]{0,2000})", s, flags=re.IGNORECASE)
    if m:
        snippet = m.group(1)
        # Extract first bracketed array inside the snippet
        arr = re.search(r"(\[\s*(?:\{[\s\S]*?\}\s*,?\s*)+\])", snippet)
        if arr:
            cases = try_parse_candidates([arr.group(1)])
            if cases:
                return json.dumps({"cases": cases, "count": len(cases)})

    return json.dumps({"cases": [], "count": 0, "note": "no test cases array found"})