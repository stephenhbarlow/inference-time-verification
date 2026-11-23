#!/usr/bin/env python
"""Run and validate the deterministic synthetic ITV example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finding_level_itv.io import (
    calculate_saved_prediction_metrics,
    load_documents,
    write_jsonl,
)
from finding_level_itv.pipeline import run_itv
from finding_level_itv.text import normalise_sentence


HERE = Path(__file__).resolve().parent


class ScriptedGenerator:
    """Return the next predefined generation for each invented document."""

    def __init__(self, generations_by_text):
        self.generations_by_text = generations_by_text
        self.next_index = {text: 0 for text in generations_by_text}

    def __call__(self, document):
        index = self.next_index[document]
        generations = self.generations_by_text[document]
        if index >= len(generations):
            raise RuntimeError("No scripted generation remains for this document")
        self.next_index[document] += 1
        return list(generations[index])


class ScriptedVerifier:
    """Return a predefined decision for each normalized invented candidate."""

    def __init__(self, decisions):
        self.decisions = decisions

    def __call__(self, candidate, _document):
        return self.decisions[normalise_sentence(candidate)]


def candidate_records(result):
    return [
        {
            "text": candidate.text,
            "normalized": candidate.normalized,
            "first_seen_generation": candidate.first_seen_generation,
            "occurrences": candidate.occurrences,
            "accepted": accepted,
        }
        for candidate, accepted in zip(result.candidates, result.decisions)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optionally write the validated predictions")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    documents = load_documents(HERE / "reports.jsonl")
    scripted = json.loads(
        (HERE / "scripted_generations.json").read_text(encoding="utf-8")
    )
    decisions = json.loads(
        (HERE / "scripted_verifier_decisions.json").read_text(encoding="utf-8")
    )
    generations_by_text = {
        document.text: scripted[document.id] for document in documents
    }
    generator = ScriptedGenerator(generations_by_text)
    verifier = ScriptedVerifier(decisions)

    predictions = []
    for document in documents:
        result = run_itv(
            document.text,
            generator,
            verifier,
            num_generations=3,
        )
        predictions.append(
            {
                "id": document.id,
                "generations": result.generations,
                "candidates": candidate_records(result),
                "findings": result.findings,
                "report_label": result.report_label,
            }
        )

    expected_predictions = [
        json.loads(line)
        for line in (HERE / "expected_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_metrics = json.loads(
        (HERE / "expected_metrics.json").read_text(encoding="utf-8")
    )
    metrics = calculate_saved_prediction_metrics(documents, predictions)
    if predictions != expected_predictions:
        raise AssertionError("Synthetic predictions differ from expected output")
    if metrics != expected_metrics:
        raise AssertionError("Synthetic metrics differ from expected output")
    if args.output:
        write_jsonl(args.output, predictions, overwrite=args.overwrite)
    print(json.dumps(metrics, indent=2))
    print("Synthetic ITV example passed.")


if __name__ == "__main__":
    main()

