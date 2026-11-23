# Input and output format

## Input JSONL

Each line contains one document:

```json
{"id": "example-001", "text": "Document text.", "findings": ["Gold finding."]}
```

`id` and `text` are required non-empty strings. IDs must be unique within a file.
`findings` is optional; when present, it must be a list of strings. Aggregate
evaluation requires findings for every document, whereas inference alone does not.

## Prediction JSONL

Each output line contains:

- `id`: the input identifier;
- `generations`: parsed finding lists from each generator call;
- `candidates`: normalized unique candidates, occurrence indices and verifier
  decisions;
- `findings`: the final returned extraction;
- `report_label`: 1 when at least one finding is returned, otherwise 0.

In generator-only mode, candidate `accepted` values are `null` because no verifier
is applied. In ITV mode, every candidate has exactly one Boolean decision.

## Aggregate metrics

When requested, the command writes one JSON object containing pooled finding
precision, recall and F1, plus document confusion counts and fixed two-class
macro-F1. Input text and individual predictions are not included in this aggregate
file.

## Verifier training JSONL

The dataset-construction command writes one contextual classification example per
line:

```json
{
  "document_id": "example-001",
  "document": "Document text.",
  "candidate": "Candidate sentence.",
  "label": 0
}
```

`label=1` identifies an existing annotated finding. `label=0` identifies an
unlabelled sentence selected as a weakly supervised negative. The latter does not
establish independent clinical incorrectness.

These JSONL files contain source documents and must remain under the same data
governance as their input. Dataset and training manifests contain configuration,
hashes and aggregate counts only; they do not contain document or candidate text.


