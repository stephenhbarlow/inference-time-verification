# Finding-level inference-time verification

This project implements finding-level inference-time verification (ITV) for
generative information extraction systems that return a variable number of
findings.

A generator is sampled repeatedly to increase candidate coverage. Findings from
all generations are normalized and de-duplicated, then a contextual verifier
accepts or rejects each finding independently while reading the complete source
document. This allows useful findings from partially correct generations to be
retained without accepting every item in those generations.

## Method overview

1. Produce `n` stochastic generations for a document.
2. Parse the finding strings from each generation.
3. Normalize and combine them into one unique candidate set.
4. Present each candidate and the complete document to the verifier.
5. Return the accepted candidates as the final variable-cardinality extraction.

The default experimental setting uses ten generations. The generator and verifier
can be LoRA adapters over the same base model, avoiding a second full set of model
weights.

## Package layout

- `candidates.py` constructs the candidate set and records generation occurrences.
- `pipeline.py` composes any compatible generator and verifier into ITV.
- `dataset.py` constructs contextual verifier examples from existing annotations.
- `sampling.py` implements uniform, Gaussian and fixed-window negative sampling.
- `prompts.py` constructs verifier prompts and parses verifier responses.
- `metrics.py` calculates pooled finding metrics and document macro-F1.
- `text.py` provides finding normalization and report sentence processing.
- `generation.py` shares prompts, parsing and decoding settings across backends.
- `backends/` provides Unsloth and Transformers model adapters.
- `training.py` validates and formats contextual verifier examples.

GPU-specific model loading and QLoRA training commands are kept separate from the
method modules so the core logic can be inspected and tested without loading a
model framework.

Use the Unsloth backend for the quantized generator/verifier workflow. Use the
Transformers backend when deterministic beam decoding is required. Both implement
the same `generate_text(messages, config)` interface, so candidate construction
and evaluation do not depend on the selected framework.

## Installation

During development, install the package and test dependencies with:

```bash
python -m pip install -e '.[text,test]'
```

Run the tests with:

```bash
pytest
```

## Inference

Copy and edit one of the configurations in `configs/`, then run:

```bash
python scripts/run_inference.py \
  --config configs/llama31_itv.json \
  --input documents.jsonl \
  --output predictions.jsonl \
  --metrics-output metrics.json
```

Omit `--metrics-output` when gold findings are unavailable. Output files are not
overwritten unless `--overwrite` is supplied. The Transformers beam-search example
is `configs/llama31_beam_huggingface.json`.

## Model-independent example

`run_itv` accepts any generator callable of the form
`generator(document) -> list[str]` and verifier callable of the form
`verifier(candidate, document) -> bool`:

```python
from finding_level_itv import run_itv

result = run_itv(
    document,
    generator=generate_findings,
    verifier=verify_finding,
    num_generations=10,
)

print(result.findings)
```

## Evaluation definitions

Finding precision, recall and F1 are calculated once from TP, FP and FN pooled
over the complete dataset. They are not averaged over reports. A report is
predicted positive when at least one finding is returned. Document F1 is the
macro-average of the positive- and negative-class F1 values.

## Data availability

Clinical reports, annotations, report-derived predictions and trained model
adapters are not distributed. The repository provides synthetic examples and
accepts user-supplied data through a documented schema. Consequently, the method
can be inspected and executed without implying that the paper's clinical results
can be reproduced without access to the governed artifacts.

See `docs/data_format.md`, `docs/backends.md`, and `docs/method.md` for the input
contract and implementation details.

## Verifier adaptation

Construct verifier examples from annotated training and validation documents:

```bash
python scripts/build_verifier_dataset.py \
  --train-input train.jsonl \
  --validation-input validation.jsonl \
  --output-dir verifier_data \
  --strategy gaussian \
  --positive-repetitions 4 \
  --negative-ratio 1.5 \
  --sigma-factor 0.2 \
  --report-prior-power 0 \
  --seed 42
```

Train the verifier adapter on a CUDA GPU:

```bash
python scripts/train_verifier.py \
  --train verifier_data/verifier_train.jsonl \
  --validation verifier_data/verifier_validation.jsonl \
  --output-dir verifier_adapter \
  --model-name '<base-model-or-local-path>' \
  --tokenizer-name meta-llama/Llama-3.1-8B-Instruct \
  --chat-template llama-3
```

The training command performs one-epoch QLoRA adaptation by default. It records
validation loss during training but does not run model generation or downstream
inference. See `docs/training.md` for the complete settings and safety boundary.

