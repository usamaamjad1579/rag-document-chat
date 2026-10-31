from typing import Protocol

from groq import Groq


class ChatModel(Protocol):
    def chat(self, messages: list[dict]) -> str: ...


class GroqChat:
    """Minimal chat wrapper so the pipeline does not depend on one SDK."""

    def __init__(self, api_key: str | None, model: str, temperature: float = 0.1):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return response.choices[0].message.content or ""
