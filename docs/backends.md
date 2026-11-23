# Model backends

The package separates model-framework operations from the ITV method. Both model
adapters receive standard chat messages and a `GenerationConfig`, and return the
decoded continuation as text.

## Unsloth

`UnslothBackend` loads a quantized model or LoRA adapter with an explicit chat
template. It converts standard system/user messages to the ShareGPT field names
used by the adapted models. Stochastic candidate generation and greedy verifier
decoding both use this backend.

## Transformers

`HuggingFaceBackend` loads either a PEFT adapter or a complete causal language
model. It uses the tokenizer's native chat template and supports sampling, greedy
decoding and deterministic beam search. Beam decoding sets `do_sample=False` and
does not pass temperature or top-p parameters.

## Separate generator and verifier backends

The generator and verifier are independent callables. They may use different
backend instances when required. For example, a beam-search generator baseline can
use Transformers, while an ITV experiment can use an Unsloth generator and
verifier.

## Optional dependencies

Install only the framework needed for a run:

```bash
python -m pip install -e '.[huggingface]'
python -m pip install -e '.[unsloth]'
```

Model and tokenizer names, adapter paths, chat templates and quantization settings
must be supplied explicitly. The package contains no private checkpoint paths.

