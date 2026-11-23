"""Unsloth backend for quantized adapter inference."""

from __future__ import annotations

from dataclasses import dataclass

from ..generation import (
    GenerationConfig,
    generation_keyword_arguments,
    terminator_token_ids,
)


@dataclass
class UnslothBackend:
    """Generate text with an Unsloth model and configurable chat template."""

    model: object
    tokenizer: object

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        chat_template: str = "llama-3",
        max_sequence_length: int = 4096,
        load_in_4bit: bool = True,
    ) -> "UnslothBackend":
        try:
            from transformers import AutoTokenizer
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError as error:  # pragma: no cover - optional GPU dependency
            raise ImportError(
                "Install finding-level-itv[unsloth] to use this backend"
            ) from error

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_sequence_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        if tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template,
            mapping={
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            map_eos_token=True,
        )
        model = FastLanguageModel.for_inference(model)
        return cls(model=model, tokenizer=tokenizer)

    def generate_text(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        sharegpt_messages = [
            {
                "from": "human" if message["role"] == "user" else "gpt",
                "value": message["content"],
            }
            for message in messages
        ]
        input_ids = self.tokenizer.apply_chat_template(
            sharegpt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model.generate(
            input_ids=input_ids,
            **generation_keyword_arguments(
                config,
                eos_token_ids=terminator_token_ids(self.tokenizer),
                pad_token_id=self.tokenizer.eos_token_id,
            ),
        )
        generated = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

