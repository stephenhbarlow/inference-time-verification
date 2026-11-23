# Commands

Training and inference commands are thin entry points over the package. Model and
data paths are supplied explicitly through configuration or command-line arguments.

`run_inference.py` supports:

- generator-only inference for single-pass or beam-search comparisons;
- ITV with independently configured generator and verifier backends;
- JSONL predictions and optional aggregate evaluation;
- non-overwriting outputs by default.

`build_verifier_dataset.py` constructs positive report--finding pairs and globally
sampled unlabelled-sentence negatives. It writes training and validation JSONL
files plus an aggregate manifest.

`train_verifier.py` trains one Unsloth QLoRA adapter on those examples. It requires
CUDA and refuses CPU-only training. It saves an adapter and training manifest but
does not run autoregressive validation or inference.


