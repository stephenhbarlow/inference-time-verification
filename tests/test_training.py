import json

import pytest

from finding_level_itv.artifacts import file_sha256
from finding_level_itv.dataset import VerifierExample
from finding_level_itv.training import (
    format_verifier_record,
    validate_verifier_jsonl,
    validate_verifier_record,
)


class RecordingTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        self.messages = messages
        return "\n".join(message["value"] for message in messages)


def test_verifier_example_has_public_training_record_fields():
    record = VerifierExample(
        document="Synthetic document.",
        candidate="Synthetic candidate.",
        label=1,
        document_id="synthetic-001",
    ).as_record()
    assert record == {
        "document_id": "synthetic-001",
        "document": "Synthetic document.",
        "candidate": "Synthetic candidate.",
        "label": 1,
    }


def test_prompt_formatting_appends_yes_or_no_completion():
    tokenizer = RecordingTokenizer()
    positive = {
        "document_id": "one",
        "document": "Synthetic document.",
        "candidate": "Synthetic candidate.",
        "label": 1,
    }
    formatted = format_verifier_record(positive, tokenizer)
    assert formatted["text"].endswith("yes")
    assert tokenizer.messages[-1] == {"from": "gpt", "value": "yes"}


def test_training_jsonl_validation_returns_only_aggregate_counts(tmp_path):
    path = tmp_path / "verifier.jsonl"
    records = [
        {
            "document_id": "one",
            "document": "Synthetic document one.",
            "candidate": "Candidate one.",
            "label": 1,
        },
        {
            "document_id": "two",
            "document": "Synthetic document two.",
            "candidate": "Candidate two.",
            "label": 0,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert validate_verifier_jsonl(path) == {
        "rows": 2,
        "positive": 1,
        "negative": 1,
    }
    assert len(file_sha256(path)) == 64


def test_boolean_is_not_accepted_as_an_integer_label():
    record = {
        "document_id": "one",
        "document": "Synthetic document.",
        "candidate": "Synthetic candidate.",
        "label": True,
    }
    with pytest.raises(ValueError, match="integer 0 or 1"):
        validate_verifier_record(record)

