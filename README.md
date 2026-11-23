# Finding-level inference-time verification

This project implements finding-level inference-time verification (ITV) for
generative information extraction systems that return a variable number of
findings.

A generator is sampled repeatedly to increase candidate coverage. Findings from
all generations are normalized and de-duplicated, then a contextual verifier
accepts or rejects each finding independently while reading the complete source
document. This allows useful findings from partially correct generations to be
retained without accepting every item in those generations.

This can be used in conjunction with https://github.com/stephenhbarlow/incidental-findings.
This provides code for the initial generator training and an earlier version of ITV.
This implementation assumes a working generator model is already trained.

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
accepts user-supplied data through a documented schema. This repository is considered a reference implementation of ITV that can be inspected and trialled without providing access to the private patient data required to generate the paper's results.

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

