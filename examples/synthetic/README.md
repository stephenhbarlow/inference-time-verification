# Synthetic ITV example

This example exercises the complete model-independent ITV path using four invented
documents. It contains no clinical records or model-generated clinical text.

The scripted generator demonstrates:

- an empty candidate pool;
- surface variants that normalize to one candidate;
- a generation containing both an accepted and a rejected candidate;
- a finding discovered only after additional sampling;
- a document with more than one accepted finding.

After installing the package, run:

```bash
python examples/synthetic/run_example.py
```

The script reconstructs every prediction, compares it with
`expected_predictions.jsonl`, recalculates the finding and document metrics, and
compares them with `expected_metrics.json`. A mismatch raises an error.

To also write the reconstructed predictions:

```bash
python examples/synthetic/run_example.py \
  --output /tmp/synthetic_predictions.jsonl
```

