# === path: astra/llms.py ===
"""LLM backends for Astra."""

from typing import Optional
import subprocess, json

class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# ---------------------------
# Default: Local Ollama
# ---------------------------
class OllamaLLM(BaseLLM):
    """Local Ollama LLM backend."""

    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            return result.stdout.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            return f"Ollama error: {e.stderr.decode('utf-8').strip()}"


# ---------------------------
# OpenAI Chat API
# ---------------------------
class OpenAIChat(BaseLLM):
    """Wrapper for OpenAI Chat models (GPT-4o, GPT-3.5, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        import openai
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ---------------------------
# Gemini Chat API
# ---------------------------
class GeminiChat(BaseLLM):
    """Wrapper for Google Gemini models."""

    def __init__(self, model: str = "gemini-pro", api_key: Optional[str] = None):
        import google.generativeai as genai
        self.model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text
