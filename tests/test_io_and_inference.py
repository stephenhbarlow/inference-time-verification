import json
from pathlib import Path

import pytest

from finding_level_itv.configuration import (
    InferenceConfig,
    ModelConfig,
    load_inference_config,
)
from finding_level_itv.generation import GenerationConfig
from finding_level_itv.inference import run_inference_documents
from finding_level_itv.io import (
    InputDocument,
    calculate_saved_prediction_metrics,
    load_documents,
    write_jsonl,
)


class ResponseBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_text(self, _messages, _config):
        return self.responses.pop(0)


def test_jsonl_contract_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        '{"id": "same", "text": "First."}\n'
        '{"id": "same", "text": "Second."}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate document id"):
        load_documents(path)


def test_jsonl_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [{"id": "one"}])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_jsonl(path, [{"id": "two"}])


def test_configuration_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "mode": "generator",
                "generator": {
                    "backend": "huggingface",
                    "model_name": "model",
                    "unexpected": 1,
                },
                "generation": {"strategy": "beam", "num_beams": 4},
                "num_generations": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown generator fields"):
        load_inference_config(path)


def test_itv_inference_uses_separate_generator_and_verifier_backends():
    generator_backend = ResponseBackend(
        [
            '{"sentences": ["Finding one."], "label": "positive"}',
            '{"sentences": ["1. Finding one."], "label": "positive"}',
        ]
    )
    verifier_backend = ResponseBackend(["yes"])

    def loader(model_config):
        return (
            generator_backend
            if model_config.model_name == "generator-model"
            else verifier_backend
        )

    config = InferenceConfig(
        mode="itv",
        generator=ModelConfig("huggingface", "generator-model"),
        verifier=ModelConfig("unsloth", "verifier-model", chat_template="llama-3"),
        generation=GenerationConfig(strategy="sample"),
        num_generations=2,
    )
    documents = [
        InputDocument("one", "Synthetic document.", ("Finding one.",))
    ]
    predictions = run_inference_documents(
        documents,
        config,
        backend_loader=loader,
    )
    assert predictions[0]["findings"] == ["Finding one."]
    assert predictions[0]["candidates"][0]["occurrences"] == [1, 2]
    assert predictions[0]["candidates"][0]["accepted"] is True
    assert calculate_saved_prediction_metrics(documents, predictions)["finding"][
        "f1"
    ] == 1.0


def test_model_examples_use_expected_native_templates():
    config_directory = Path(__file__).parents[1] / "configs"
    expected = {
        "llama31_itv.json": "llama-3",
        "llama32_itv.json": "llama-3",
        "mistral_itv.json": "mistral",
        "gemma2_itv.json": "gemma2",
        "phi4_itv.json": "phi-4",
    }
    for filename, template in expected.items():
        config = json.loads((config_directory / filename).read_text(encoding="utf-8"))
        assert config["generator"]["chat_template"] == template
        assert config["verifier"]["chat_template"] == template
    beam = json.loads(
        (config_directory / "llama31_beam_huggingface.json").read_text(
            encoding="utf-8"
        )
    )
    assert beam["generator"]["backend"] == "huggingface"
    assert beam["generation"]["strategy"] == "beam"
    assert beam["generation"]["num_beams"] == 4

