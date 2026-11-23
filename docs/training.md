# Verifier dataset construction and adaptation

## Sentence preparation

The construction command tokenizes each document into ordered sentences and aligns
annotations by exact sentence match, followed by substring containment when an
annotation is embedded in a longer sentence. Aligned annotation sentences are
removed from negative eligibility.

Default sentence splitting uses NLTK. Install the package extra and tokenizer data
before constructing a dataset:

```bash
python -m pip install -e '.[text]'
python -m nltk.downloader punkt punkt_tab
```

## Training examples

Annotated findings form positive report--candidate pairs. Positive examples are
repeated four times by default. Eligible unlabelled sentences receive either
uniform, Gaussian or fixed-window position weights. Reports receive equal initial
mass when `--report-prior-power 0` is used. Negatives are then selected globally by
a seeded probability-proportional exponential race, without replacement and with
normalized sentence-string de-duplication across documents.

The default final ratio is 1.5 negatives per repeated positive. Validation uses
one copy of each positive and a uniform 1:1 negative construction. The dataset
manifest records all settings, input/output hashes and aggregate class counts.

## QLoRA settings

`train_verifier.py` defaults to the settings used by the project:

- one epoch;
- 4-bit model loading;
- LoRA rank 16 and alpha 64;
- LoRA dropout 0;
- batch size 1;
- gradient accumulation over 8 steps;
- learning rate `2e-4`;
- 10 warm-up steps;
- linear learning-rate schedule;
- 8-bit AdamW;
- maximum sequence length 4096;
- seed 42.

LoRA targets the query, key, value, output, gate, up and down projection modules.
Training uses the complete serialized report--candidate conversation and yes/no
completion. Epoch-level validation loss is calculated, but no validation examples
are decoded and no extraction inference is started automatically.

The command refuses to run without CUDA and refuses to overwrite an existing
output directory. Its manifest records configuration, software versions, GPU name,
input hashes, duration and adapter hashes without including training text.

