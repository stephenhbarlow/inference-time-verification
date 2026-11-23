from dataclasses import dataclass, field

import pytest

from finding_level_itv.generation import (
    FindingGenerator,
    FindingVerifier,
    GenerationConfig,
    generation_keyword_arguments,
    parse_generator_response,
    terminator_token_ids,
)


@dataclass
class StubBackend:
    responses: list[str]
    calls: list[tuple[list[dict[str, str]], GenerationConfig]] = field(
        default_factory=list
    )

    def generate_text(self, messages, config):
        self.calls.append((messages, config))
        return self.responses.pop(0)


class StubTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"<|eot_id|>": 3, "<|im_end|>": 4}


def test_sampling_and_beam_arguments_are_kept_separate():
    sampled = generation_keyword_arguments(
        GenerationConfig(strategy="sample", temperature=0.5, top_p=0.5),
        eos_token_ids=[2],
        pad_token_id=2,
    )
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.5
    assert "num_beams" not in sampled

    beam = generation_keyword_arguments(
        GenerationConfig(strategy="beam", num_beams=8),
        eos_token_ids=[2],
        pad_token_id=2,
    )
    assert beam["do_sample"] is False
    assert beam["num_beams"] == 8
    assert "temperature" not in beam


def test_greedy_verification_has_no_sampling_or_beam_parameters():
    arguments = generation_keyword_arguments(
        GenerationConfig(strategy="greedy", max_new_tokens=10),
        eos_token_ids=[2],
        pad_token_id=2,
    )
    assert arguments["do_sample"] is False
    assert "temperature" not in arguments
    assert "num_beams" not in arguments


def test_terminators_include_only_tokens_present_in_vocabulary():
    assert terminator_token_ids(StubTokenizer()) == [2, 3, 4]


def test_generator_response_allows_surrounding_model_text():
    assert parse_generator_response(
        'Answer:\n{"sentences": ["Finding one."], "label": "positive"}\nDone.'
    ) == ["Finding one."]
    with pytest.raises(ValueError, match="positive or negative"):
        parse_generator_response('{"sentences": [], "label": "unknown"}')


def test_invalid_generator_output_is_retried_with_sampling():
    backend = StubBackend(
        responses=[
            "not json",
            "still not json",
            '{"sentences": ["Finding one."], "label": "positive"}',
        ]
    )
    generator = FindingGenerator(
        backend,
        GenerationConfig(strategy="beam", num_beams=4),
        max_parse_attempts=3,
    )
    assert generator("Synthetic document.") == ["Finding one."]
    assert backend.calls[0][1].strategy == "beam"
    assert backend.calls[1][1].strategy == "beam"
    assert backend.calls[2][1].strategy == "sample"


def test_verifier_uses_greedy_decoding_and_response_parser():
    backend = StubBackend(responses=["yes"])
    verifier = FindingVerifier(backend)
    assert verifier("Candidate.", "Synthetic document.") is True
    assert backend.calls[0][1].strategy == "greedy"

