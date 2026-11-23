"""Inference configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .generation import GenerationConfig


BackendName = Literal["unsloth", "huggingface"]
InferenceMode = Literal["generator", "itv"]


@dataclass(frozen=True)
class ModelConfig:
    """Model location and framework-specific loading settings."""

    backend: BackendName
    model_name: str
    tokenizer_name: str | None = None
    chat_template: str | None = None
    adapter: bool = True
    load_in_4bit: bool = True
    max_sequence_length: int = 4096

    def validate(self, name: str) -> None:
        if self.backend not in ("unsloth", "huggingface"):
            raise ValueError(f"{name}.backend must be unsloth or huggingface")
        if not self.model_name or self.model_name.startswith("<"):
            raise ValueError(f"{name}.model_name must be replaced with a model path")
        if self.max_sequence_length < 1:
            raise ValueError(f"{name}.max_sequence_length must be positive")
        if self.backend == "unsloth" and not self.chat_template:
            raise ValueError(f"{name}.chat_template is required for Unsloth")


@dataclass(frozen=True)
class InferenceConfig:
    """Complete generator-only or ITV inference configuration."""

    mode: InferenceMode
    generator: ModelConfig
    generation: GenerationConfig
    verifier: ModelConfig | None = None
    num_generations: int = 10
    max_parse_attempts: int = 2
    seed: int = 42

    def validate(self) -> None:
        if self.mode not in ("generator", "itv"):
            raise ValueError("mode must be generator or itv")
        self.generator.validate("generator")
        self.generation.validate()
        if self.mode == "itv":
            if self.verifier is None:
                raise ValueError("ITV mode requires a verifier configuration")
            self.verifier.validate("verifier")
        if self.num_generations < 1:
            raise ValueError("num_generations must be positive")
        if self.mode == "generator" and self.num_generations != 1:
            raise ValueError("generator mode requires num_generations=1")
        if self.max_parse_attempts < 1:
            raise ValueError("max_parse_attempts must be positive")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _model_config(value: Any, name: str) -> ModelConfig:
    mapping = _require_mapping(value, name)
    allowed = {
        "backend",
        "model_name",
        "tokenizer_name",
        "chat_template",
        "adapter",
        "load_in_4bit",
        "max_sequence_length",
    }
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return ModelConfig(**mapping)


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load a strict JSON configuration and reject misspelled fields."""
    mapping = _require_mapping(
        json.loads(Path(path).read_text(encoding="utf-8")), "configuration"
    )
    allowed = {
        "mode",
        "generator",
        "generation",
        "verifier",
        "num_generations",
        "max_parse_attempts",
        "seed",
    }
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    generation_mapping = _require_mapping(mapping.get("generation"), "generation")
    generation_allowed = {
        "strategy",
        "temperature",
        "top_p",
        "num_beams",
        "max_new_tokens",
        "max_time",
        "early_stopping",
    }
    generation_unknown = set(generation_mapping) - generation_allowed
    if generation_unknown:
        raise ValueError(
            f"Unknown generation fields: {sorted(generation_unknown)}"
        )
    config = InferenceConfig(
        mode=mapping.get("mode"),
        generator=_model_config(mapping.get("generator"), "generator"),
        verifier=(
            _model_config(mapping["verifier"], "verifier")
            if mapping.get("verifier") is not None
            else None
        ),
        generation=GenerationConfig(**generation_mapping),
        num_generations=mapping.get("num_generations", 10),
        max_parse_attempts=mapping.get("max_parse_attempts", 2),
        seed=mapping.get("seed", 42),
    )
    config.validate()
    return config

