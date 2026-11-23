"""Document-level orchestration shared by the inference command and examples."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .backends import HuggingFaceBackend, UnslothBackend
from .candidates import build_candidate_pool
from .configuration import InferenceConfig, ModelConfig
from .generation import FindingGenerator, FindingVerifier, TextGenerationBackend
from .io import InputDocument
from .pipeline import run_itv


BackendLoader = Callable[[ModelConfig], TextGenerationBackend]


def load_model_backend(config: ModelConfig) -> TextGenerationBackend:
    """Instantiate the configured optional model-framework adapter."""
    if config.backend == "huggingface":
        return HuggingFaceBackend.load(
            config.model_name,
            tokenizer_name=config.tokenizer_name,
            adapter=config.adapter,
            load_in_4bit=config.load_in_4bit,
        )
    if config.backend == "unsloth":
        return UnslothBackend.load(
            config.model_name,
            tokenizer_name=config.tokenizer_name,
            chat_template=config.chat_template or "llama-3",
            max_sequence_length=config.max_sequence_length,
            load_in_4bit=config.load_in_4bit,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def set_inference_seed(seed: int) -> None:
    """Seed available random-number generators without requiring model packages."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def _candidate_records(candidates, decisions: list[bool] | None) -> list[dict[str, Any]]:
    records = []
    for index, candidate in enumerate(candidates):
        records.append(
            {
                "text": candidate.text,
                "normalized": candidate.normalized,
                "first_seen_generation": candidate.first_seen_generation,
                "occurrences": candidate.occurrences,
                "accepted": decisions[index] if decisions is not None else None,
            }
        )
    return records


def run_inference_documents(
    documents: list[InputDocument],
    config: InferenceConfig,
    *,
    backend_loader: BackendLoader = load_model_backend,
) -> list[dict[str, Any]]:
    """Run generator-only or ITV inference and return JSON-serializable records."""
    config.validate()
    set_inference_seed(config.seed)
    generator = FindingGenerator(
        backend_loader(config.generator),
        config.generation,
        max_parse_attempts=config.max_parse_attempts,
    )
    verifier = None
    if config.mode == "itv":
        if config.verifier is None:  # guarded by validate; keeps typing explicit
            raise ValueError("ITV mode requires a verifier")
        verifier = FindingVerifier(backend_loader(config.verifier))

    records: list[dict[str, Any]] = []
    for document in documents:
        if config.mode == "itv":
            if verifier is None:
                raise AssertionError("validated ITV configuration has no verifier")
            result = run_itv(
                document.text,
                generator,
                verifier,
                num_generations=config.num_generations,
            )
            generations = result.generations
            candidates = result.candidates
            decisions: list[bool] | None = result.decisions
            findings = result.findings
        else:
            generated = generator(document.text)
            generations = [generated]
            candidates = build_candidate_pool(generations)
            decisions = None
            # Generator-only evaluation removes exact duplicate strings but does
            # not apply verifier candidate filtering.
            findings = list(dict.fromkeys(generated))

        records.append(
            {
                "id": document.id,
                "generations": generations,
                "candidates": _candidate_records(candidates, decisions),
                "findings": findings,
                "report_label": int(bool(findings)),
            }
        )
    return records

