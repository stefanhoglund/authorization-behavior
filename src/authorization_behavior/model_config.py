from dataclasses import dataclass
from pathlib import Path

import yaml

from authorization_behavior.models import (
    ModelClient,
    OllamaClient,
)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    backend: str
    model: str
    group: str | None = None


def load_model_configs(
    path: Path,
) -> list[ModelConfig]:

    with path.open() as f:
        data = yaml.safe_load(f)

    return [ModelConfig(**item) for item in data["models"]]


def create_model_client(
    config: ModelConfig,
    temperature: float,
) -> ModelClient:

    if config.backend == "ollama":
        return OllamaClient(
            model_id=config.id,
            model_name=config.model,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported backend: {config.backend}")
