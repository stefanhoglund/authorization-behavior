import time
from dataclasses import dataclass
from typing import Protocol

import ollama


@dataclass(frozen=True)
class ModelResponse:
    text: str
    latency_seconds: float


class ModelClient(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        prompt: str,
    ) -> ModelResponse: ...


class OllamaClient:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        temperature: float = 0.0,
    ):
        self._model_id = model_id
        self._model_name = model_name
        self.temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
    ) -> ModelResponse:
        start = time.perf_counter()

        response = ollama.chat(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": self.temperature,
            },
        )

        elapsed = time.perf_counter() - start

        return ModelResponse(
            text=response["message"]["content"],
            latency_seconds=elapsed,
        )


# import time
# from dataclasses import dataclass

# import ollama


# @dataclass(frozen=True)
# class ModelResponse:
#     text: str
#     latency_seconds: float


# class OllamaClient:
#     def __init__(
#         self,
#         model_id: str,
#         model_name: str,
#         temperature: float = 0.0,
#     ):
#         self._model_id = model_id
#         self._model_name = model_name
#         self.temperature = temperature

#     @property
#     def model_id(self) -> str:
#         return self._model_id

#     @property
#     def model_name(self) -> str:
#         return self._model_name

#     def generate(
#         self,
#         prompt: str,
#     ) -> ModelResponse:
#         start = time.perf_counter()

#         response = ollama.chat(
#             model=self._model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             options={
#                 "temperature": self.temperature,
#             },
#         )

#         elapsed = time.perf_counter() - start

#         return ModelResponse(
#             text=response["message"]["content"],
#             latency_seconds=elapsed,
#         )


# # # src/authorization_behavior/models.py

# # from dataclasses import dataclass
# # from typing import Protocol


# # @dataclass
# # class ModelResponse:
# #     text: str
# #     model: str
# #     latency_seconds: float


# # class ModelClient(Protocol):
# #     @property
# #     def model_id(self) -> str: ...

# #     def generate(
# #         self,
# #         prompt: str,
# #     ) -> ModelResponse: ...


# # class OllamaClient:
# #     def __init__(
# #         self,
# #         model_id: str,
# #         model_name: str,
# #         temperature: float = 0.0,
# #     ):
# #         self._model_id = model_id
# #         self._model_name = model_name
# #         self.temperature = temperature

# #     @property
# #     def model_id(self) -> str:
# #         return self._model_id

# #     @property
# #     def model_name(self) -> str:
# #         return self._model_name


# # # class OllamaClient:
# # #     def __init__(
# # #         self,
# # #         model: str,
# # #         temperature: float = 0.0,
# # #     ):
# # #         self.model = model
# # #         self.temperature = temperature

# # #     @property
# # #     def model_id(self) -> str:
# # #         return self.model

# # #     def generate(
# # #         self,
# # #         prompt: str,
# # #     ) -> ModelResponse:
# # #         start = time.perf_counter()

# # #         response = ollama.chat(
# # #             model=self.model,
# # #             messages=[
# # #                 {
# # #                     "role": "user",
# # #                     "content": prompt,
# # #                 }
# # #             ],
# # #             options={
# # #                 "temperature": self.temperature,
# # #             },
# # #         )

# # #         latency = time.perf_counter() - start

# # #         return ModelResponse(
# # #             text=response["message"]["content"],
# # #             model=self.model,
# # #             latency_seconds=latency,
# # #         )


# # # from dataclasses import dataclass
# # # from typing import Protocol
# # # from typing import Any
# # # from time import perf_counter
# # # from ollama import chat

# # # @dataclass
# # # class ModelResponse:
# # #     text: str
# # #     model: str
# # #     thinking: str | None = None
# # #     latency_seconds: float | None = None
# # #     raw: Any | None = None

# # # class ModelClient(Protocol):
# # #     def generate(
# # #         self,
# # #         prompt:str,
# # #         system_prompt: str | None = None,
# # #         ) -> ModelResponse:
# # #         ...

# # # class OllamaClient:
# # #     def __init__(
# # #         self,
# # #         model: str = "qwen3:4b-instruct-2507-q4_K_M",
# # #         temperature: float = 0.0,
# # #         think: bool = False,
# # #     ):
# # #         self.model = model
# # #         self.temperature = temperature
# # #         self.think = think

# # #     def generate(self,
# # #         prompt: str,
# # #         system_prompt: str | None = None
# # #         ) -> ModelResponse:

# # #         messages = []

# # #         if system_prompt is not None:
# # #             messages.append(
# # #                 {
# # #                     "role": "system",
# # #                     "content": system_prompt,
# # #                 }
# # #             )

# # #         messages.append(
# # #             {
# # #                 "role": "user",
# # #                 "content": prompt,
# # #             }
# # #         )

# # #         start = perf_counter()

# # #         response = chat(
# # #             model = self.model,
# # #             messages = messages,
# # #             # think = self.think,
# # #             options = {
# # #                 "temperature": self.temperature,
# # #             },
# # #         )

# # #         latency = perf_counter() - start

# # #         return ModelResponse(
# # #             text = response.message.content.strip(),
# # #             thinking = getattr(response.message, "thinking", None),
# # #             model = self.model,
# # #             latency_seconds = latency,
# # #             raw = response,
# # #         )
