"""Transformers backend, including deterministic beam-search decoding."""

from __future__ import annotations

from dataclasses import dataclass

from ..generation import (
    GenerationConfig,
    generation_keyword_arguments,
    terminator_token_ids,
)


@dataclass
class HuggingFaceBackend:
    """Generate text with a Transformers model and its native chat template."""

    model: object
    tokenizer: object

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        adapter: bool = True,
        load_in_4bit: bool = True,
    ) -> "HuggingFaceBackend":
        """Load either a PEFT adapter or a complete causal language model."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:  # pragma: no cover - optional GPU dependency
            raise ImportError(
                "Install finding-level-itv[huggingface] to use this backend"
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = (
            BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        )
        load_arguments = {"device_map": "auto"}
        if quantization is not None:
            load_arguments["quantization_config"] = quantization
        if adapter:
            try:
                from peft import AutoPeftModelForCausalLM
            except ImportError as error:  # pragma: no cover
                raise ImportError(
                    "PEFT is required when loading an adapter checkpoint"
                ) from error
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name, **load_arguments
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_arguments
            )
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    def generate_text(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        outputs = self.model.generate(
            input_ids=input_ids,
            **generation_keyword_arguments(
                config,
                eos_token_ids=terminator_token_ids(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        generated = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

