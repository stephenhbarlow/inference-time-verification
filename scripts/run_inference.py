#!/usr/bin/env python
"""Run generator-only or finding-level ITV inference from a JSON configuration."""

from __future__ import annotations

import argparse
import json

from finding_level_itv.configuration import load_inference_config
from finding_level_itv.inference import run_inference_documents
from finding_level_itv.io import (
    calculate_saved_prediction_metrics,
    load_documents,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference configuration JSON")
    parser.add_argument("--input", required=True, help="Input document JSONL")
    parser.add_argument("--output", required=True, help="Prediction JSONL")
    parser.add_argument(
        "--metrics-output",
        help="Optional aggregate JSON; requires findings for every input document",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of output files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_inference_config(args.config)
    documents = load_documents(args.input)
    predictions = run_inference_documents(documents, config)
    write_jsonl(args.output, predictions, overwrite=args.overwrite)
    if args.metrics_output:
        metrics = calculate_saved_prediction_metrics(documents, predictions)
        write_json(args.metrics_output, metrics, overwrite=args.overwrite)
        print(json.dumps(metrics, indent=2))
    print(f"Wrote {len(predictions)} prediction records to {args.output}")


if __name__ == "__main__":
    main()

