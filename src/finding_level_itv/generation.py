"""Shared prompts, parsing and decoding settings for finding generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from string import punctuation
from typing import Literal, Protocol


GenerationStrategy = Literal["greedy", "sample", "beam"]


@dataclass(frozen=True)
class GenerationConfig:
    """Backend-independent decoding settings."""

    strategy: GenerationStrategy = "sample"
    temperature: float = 0.5
    top_p: float = 0.5
    num_beams: int = 4
    max_new_tokens: int | None = None
    max_time: float | None = 60.0
    early_stopping: bool = True

    def validate(self) -> None:
        if self.strategy not in ("greedy", "sample", "beam"):
            raise ValueError("strategy must be 'greedy', 'sample' or 'beam'")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.strategy == "sample":
            if self.temperature <= 0:
                raise ValueError("temperature must be positive")
            if not 0 < self.top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
        if self.strategy == "beam" and self.num_beams < 2:
            raise ValueError("beam decoding requires num_beams >= 2")


class TextGenerationBackend(Protocol):
    """Interface implemented by Unsloth and Transformers model adapters."""

    def generate_text(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> str: ...


def generator_messages(document: str) -> list[dict[str, str]]:
    """Build the report-to-findings instruction using standard chat roles."""
    schema = (
        "The output should be a markdown code snippet formatted in the following "
        'json schema: {"sentences": list of strings // a list of the actionable '
        "incidental findings as strings, or an empty list if there are no "
        'actionable incidental findings in the report., "label": string // '
        '"positive" if there are any actionable incidental findings in the '
        'report, or "negative" only.}'
    )
    return [
        {
            "role": "system",
            "content": "The following text is a PET-CT report for lung cancer:",
        },
        {
            "role": "user",
            "content": (
                f"REPORT: {document.strip().lstrip(punctuation).strip()}\n\n"
                "INSTRUCTION: Extract the sentences in the report indicating "
                "actionable incidental findings requiring medical intervention, "
                'then label the overall report "positive" (if there are any '
                'actionable incidental findings in the report), or "negative".'
                f"\n\n{schema}"
            ),
        },
    ]


def verifier_messages(candidate: str, document: str) -> list[dict[str, str]]:
    """Build a candidate-in-document verification prompt with standard roles."""
    return [
        {
            "role": "system",
            "content": "The following text is a PET-CT report for lung cancer:",
        },
        {
            "role": "user",
            "content": (
                f"REPORT: {document}\n\n"
                "INSTRUCTION: In the context of the report would the following "
                "sentence be related to an actionable incidental finding?\n\n"
                f'"{candidate}"\n\n'
                "Answer 'yes' or 'no' only."
            ),
        },
    ]


def generation_keyword_arguments(
    config: GenerationConfig,
    *,
    eos_token_ids: list[int],
    pad_token_id: int | None,
) -> dict[str, object]:
    """Translate shared decoding settings into ``model.generate`` arguments."""
    config.validate()
    arguments: dict[str, object] = {
        "do_sample": config.strategy == "sample",
        "eos_token_id": eos_token_ids,
        "pad_token_id": pad_token_id,
    }
    if config.max_new_tokens is not None:
        arguments["max_new_tokens"] = config.max_new_tokens
    if config.max_time is not None:
        arguments["max_time"] = config.max_time
    if config.strategy == "sample":
        arguments.update(temperature=config.temperature, top_p=config.top_p)
    elif config.strategy == "beam":
        arguments.update(
            num_beams=config.num_beams,
            early_stopping=config.early_stopping,
        )
    return arguments


def terminator_token_ids(tokenizer: object) -> list[int]:
    """Return supported end-of-sequence and end-of-turn token IDs."""
    token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
    vocabulary = tokenizer.get_vocab()
    for token in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        if token in vocabulary:
            token_ids.append(int(vocabulary[token]))
    return list(dict.fromkeys(token_ids))


def parse_generator_response(response: str) -> list[str]:
    """Parse and validate the JSON finding list from a model response."""
    start = response.find("{")
    end = response.find("}", start + 1)
    if start < 0 or end < 0:
        raise ValueError("generator response does not contain a JSON object")
    try:
        payload = json.loads(response[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("generator response contains invalid JSON") from error
    sentences = payload.get("sentences")
    label = payload.get("label")
    if not isinstance(sentences, list) or any(
        not isinstance(sentence, str) for sentence in sentences
    ):
        raise ValueError("generator JSON requires a list of string sentences")
    if label not in ("positive", "negative"):
        raise ValueError("generator JSON requires a positive or negative label")
    return sentences


@dataclass
class FindingGenerator:
    """Callable adapter that prompts a text backend and validates its JSON."""

    backend: TextGenerationBackend
    config: GenerationConfig = GenerationConfig()
    max_parse_attempts: int = 2

    def __call__(self, document: str) -> list[str]:
        if self.max_parse_attempts < 1:
            raise ValueError("max_parse_attempts must be positive")
        config = self.config
        for attempt in range(self.max_parse_attempts):
            response = self.backend.generate_text(generator_messages(document), config)
            try:
                return parse_generator_response(response)
            except ValueError:
                if attempt + 1 == self.max_parse_attempts:
                    return []
                # Keep the first retry under the requested strategy. If further
                # attempts are allowed, switch subsequent attempts to sampling.
                if attempt >= 1:
                    config = replace(
                        config,
                        strategy="sample",
                        temperature=0.5,
                        top_p=0.5,
                    )
        raise AssertionError("unreachable")


@dataclass
class FindingVerifier:
    """Callable adapter that converts a verifier response into a Boolean decision."""

    backend: TextGenerationBackend
    max_new_tokens: int = 10

    def __call__(self, candidate: str, document: str) -> bool:
        from .prompts import verifier_accepts_response

        response = self.backend.generate_text(
            verifier_messages(candidate, document),
            GenerationConfig(
                strategy="greedy",
                max_new_tokens=self.max_new_tokens,
                max_time=None,
            ),
        )
        return verifier_accepts_response(response)

