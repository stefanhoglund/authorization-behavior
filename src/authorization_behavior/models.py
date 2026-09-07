
from dataclasses import dataclass
from typing import Protocol
from typing import Any
from time import perf_counter
from ollama import chat

@dataclass
class ModelResponse:
    text: str
    model: str
    thinking: str | None = None
    latency_seconds: float | None = None
    raw: Any | None = None

class ModelClient(Protocol):
    def generate(
        self,
        prompt:str,
        system_prompt: str | None = None,
        ) -> ModelResponse:
        ...

class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:4b-instruct-2507-q4_K_M",
        temperature: float = 0.0,
        think: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.think = think

    def generate(self,
        prompt: str,
        system_prompt: str | None = None
        ) -> ModelResponse:
        
        messages = []
        
        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        start = perf_counter()

        response = chat(
            model = self.model,
            messages = messages,
            # think = self.think,
            options = {
                "temperature": self.temperature,
            },
        )

        latency = perf_counter() - start

        return ModelResponse(
            text = response.message.content.strip(),
            thinking = getattr(response.message, "thinking", None),
            model = self.model,
            latency_seconds = latency,
            raw = response,
        )
