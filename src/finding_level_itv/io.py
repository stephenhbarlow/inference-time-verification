"""Validated JSONL input and output for inference commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    calculate_document_metrics,
    calculate_pooled_extraction_metrics,
)


@dataclass(frozen=True)
class InputDocument:
    """One inference document with optional gold findings."""

    id: str
    text: str
    findings: tuple[str, ...] | None = None


def load_documents(path: str | Path) -> list[InputDocument]:
    """Read and validate UTF-8 JSONL documents."""
    source = Path(path)
    documents: list[InputDocument] = []
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            document_id = record.get("id")
            text = record.get("text")
            findings = record.get("findings")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"Line {line_number} requires a non-empty string id")
            if document_id in seen_ids:
                raise ValueError(f"Duplicate document id: {document_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Document {document_id} requires non-empty text")
            if findings is not None and (
                not isinstance(findings, list)
                or any(not isinstance(item, str) for item in findings)
            ):
                raise ValueError(
                    f"Document {document_id} findings must be a list of strings"
                )
            seen_ids.add(document_id)
            documents.append(
                InputDocument(
                    id=document_id,
                    text=text,
                    findings=tuple(findings) if findings is not None else None,
                )
            )
    if not documents:
        raise ValueError("Input JSONL contains no documents")
    return documents


def write_jsonl(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    """Write records atomically and refuse accidental replacement by default."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def write_json(
    path: str | Path,
    value: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write one formatted JSON object atomically."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)



def calculate_saved_prediction_metrics(
    documents: list[InputDocument],
    prediction_records: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    """Calculate finding and document metrics for fully annotated input."""
    if len(documents) != len(prediction_records):
        raise ValueError("Documents and predictions must align")
    if any(document.findings is None for document in documents):
        raise ValueError("Metrics require findings for every input document")
    predicted = [record["findings"] for record in prediction_records]
    gold = [list(document.findings or ()) for document in documents]
    finding_metrics = calculate_pooled_extraction_metrics(predicted, gold)
    document_pairs = [
        (int(bool(document.findings)), int(bool(record["findings"])))
        for document, record in zip(documents, prediction_records)
    ]
    return {
        "finding": finding_metrics,
        "document": calculate_document_metrics(document_pairs),
    }

