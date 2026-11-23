#!/usr/bin/env python
"""Construct weakly supervised verifier training and validation JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from finding_level_itv.artifacts import file_sha256, utc_now
from finding_level_itv.dataset import (
    AnnotatedReport,
    NegativeConstructionConfig,
    build_verifier_training_examples,
)
from finding_level_itv.io import load_documents, write_json, write_jsonl
from finding_level_itv.text import split_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--validation-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strategy",
        choices=("uniform", "gaussian", "central_window"),
        default="gaussian",
    )
    parser.add_argument("--positive-repetitions", type=int, default=4)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--sigma-factor", type=float, default=0.2)
    parser.add_argument("--central-window-fraction", type=float, default=0.4)
    parser.add_argument("--report-prior-power", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_annotated_reports(path: Path) -> list[AnnotatedReport]:
    documents = load_documents(path)
    if any(document.findings is None for document in documents):
        raise ValueError(f"Every document in {path} requires findings")
    reports = []
    for document in documents:
        try:
            sentences = tuple(split_report(document.text))
        except LookupError as error:
            raise RuntimeError(
                "NLTK sentence-tokenizer data is unavailable. Install the NLTK "
                "punkt resources before constructing verifier datasets."
            ) from error
        reports.append(
            AnnotatedReport(
                id=document.id,
                text=document.text,
                sentences=sentences,
                findings=tuple(document.findings or ()),
            )
        )
    return reports


def label_counts(records: list[dict[str, str | int]]) -> dict[str, int]:
    counts = Counter(record["label"] for record in records)
    return {
        "rows": len(records),
        "positive": counts[1],
        "negative": counts[0],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_output = output_dir / "verifier_train.jsonl"
    validation_output = output_dir / "verifier_validation.jsonl"
    manifest_output = output_dir / "dataset_manifest.json"
    for path in (train_output, validation_output, manifest_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    train_reports = to_annotated_reports(Path(args.train_input))
    validation_reports = to_annotated_reports(Path(args.validation_input))
    training_config = NegativeConstructionConfig(
        strategy=args.strategy,
        positive_repetitions=args.positive_repetitions,
        negative_ratio=args.negative_ratio,
        sigma_factor=args.sigma_factor,
        central_window_fraction=args.central_window_fraction,
        report_prior_power=args.report_prior_power,
        seed=args.seed,
    )
    validation_config = NegativeConstructionConfig(
        strategy="uniform",
        positive_repetitions=1,
        negative_ratio=1.0,
        report_prior_power=0.0,
        seed=args.seed,
    )
    train_records = [
        example.as_record()
        for example in build_verifier_training_examples(
            train_reports, training_config
        )
    ]
    validation_records = [
        example.as_record()
        for example in build_verifier_training_examples(
            validation_reports, validation_config
        )
    ]
    write_jsonl(train_output, train_records)
    write_jsonl(validation_output, validation_records)
    write_json(
        manifest_output,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "seed": args.seed,
            "training": {
                "strategy": args.strategy,
                "positive_repetitions": args.positive_repetitions,
                "negative_ratio": args.negative_ratio,
                "sigma_factor": args.sigma_factor,
                "central_window_fraction": args.central_window_fraction,
                "report_prior_power": args.report_prior_power,
                "global_normalized_sentence_deduplication": True,
                "counts": label_counts(train_records),
            },
            "validation": {
                "strategy": "uniform",
                "positive_repetitions": 1,
                "negative_ratio": 1.0,
                "counts": label_counts(validation_records),
            },
            "inputs": {
                "train_path": str(args.train_input),
                "train_sha256": file_sha256(args.train_input),
                "validation_path": str(args.validation_input),
                "validation_sha256": file_sha256(args.validation_input),
            },
            "outputs": {
                "train_path": str(train_output),
                "train_sha256": file_sha256(train_output),
                "validation_path": str(validation_output),
                "validation_sha256": file_sha256(validation_output),
            },
            "note": "Manifest contains configuration, hashes and aggregate counts only.",
        },
    )
    print(f"Wrote {len(train_records)} training examples to {train_output}")
    print(
        f"Wrote {len(validation_records)} validation examples to {validation_output}"
    )


if __name__ == "__main__":
    main()

