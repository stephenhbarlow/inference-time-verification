"""Validation and prompt formatting for contextual verifier adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from string import punctuation
from typing import Any

from .prompts import sharegpt_verifier_messages


REQUIRED_TRAINING_FIELDS = {"document_id", "document", "candidate", "label"}


def validate_verifier_record(record: Any, context: str = "record") -> None:
    """Validate one verifier training record without exposing its contents."""
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = REQUIRED_TRAINING_FIELDS - set(record)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if not isinstance(record["document_id"], str):
        raise ValueError(f"{context}.document_id must be a string")
    if not isinstance(record["document"], str) or not record["document"].strip():
        raise ValueError(f"{context}.document must be non-empty text")
    if not isinstance(record["candidate"], str) or not record["candidate"].strip():
        raise ValueError(f"{context}.candidate must be non-empty text")
    if record["label"] not in (0, 1) or isinstance(record["label"], bool):
        raise ValueError(f"{context}.label must be integer 0 or 1")


def validate_verifier_jsonl(path: str | Path) -> dict[str, int]:
    """Validate a complete JSONL file and return aggregate label counts."""
    counts = {"rows": 0, "positive": 0, "negative": 0}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            validate_verifier_record(record, f"line {line_number}")
            counts["rows"] += 1
            if record["label"] == 1:
                counts["positive"] += 1
            else:
                counts["negative"] += 1
    if counts["rows"] == 0:
        raise ValueError("Verifier JSONL contains no records")
    return counts


def format_verifier_record(record: dict[str, Any], tokenizer: object) -> dict[str, Any]:
    """Serialize one report--candidate pair and its yes/no completion."""
    validate_verifier_record(record)
    document = record["document"].strip().lstrip(punctuation).strip()
    messages = sharegpt_verifier_messages(record["candidate"], document)
    messages.append(
        {"from": "gpt", "value": "yes" if record["label"] == 1 else "no"}
    )
    return {
        **record,
        "text": tokenizer.apply_chat_template(messages, tokenize=False),
    }

