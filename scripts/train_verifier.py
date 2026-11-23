#!/usr/bin/env python
"""Train one contextual verifier LoRA adapter with Unsloth."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from finding_level_itv.artifacts import (
    file_sha256,
    first_existing_sha256,
    package_versions,
    utc_now,
)
from finding_level_itv.io import write_json
from finding_level_itv.training import (
    format_verifier_record,
    validate_verifier_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training JSONL")
    parser.add_argument("--validation", required=True, help="Validation JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "max_sequence_length",
        "lora_rank",
        "lora_alpha",
        "batch_size",
        "gradient_accumulation_steps",
        "save_total_limit",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if args.warmup_steps < 0 or args.lora_dropout < 0:
        raise ValueError("warmup steps and LoRA dropout cannot be negative")


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    for path in (train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_counts = validate_verifier_jsonl(train_path)
    validation_counts = validate_verifier_jsonl(validation_path)
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "training_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "initialising",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "configuration": vars(args),
        "inputs": {
            "train_path": str(train_path),
            "train_sha256": file_sha256(train_path),
            "train_counts": train_counts,
            "validation_path": str(validation_path),
            "validation_sha256": file_sha256(validation_path),
            "validation_counts": validation_counts,
        },
        "software": package_versions(
            ("torch", "transformers", "datasets", "peft", "trl", "unsloth")
        ),
        "note": (
            "The command trains and records validation loss only; it does not "
            "run autoregressive validation or downstream inference."
        ),
    }
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        # GPU libraries are imported only after paths and JSONL inputs pass their
        # inexpensive validation checks.
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing CPU-only verifier training")

        # Unsloth must be imported before Transformers and TRL so its model and
        # trainer patches are applied before those classes are loaded.
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        manifest.update(
            {
                "status": "running",
                "updated_utc": utc_now(),
                "hardware": {
                    "gpu_name": torch.cuda.get_device_name(),
                    "torch_cuda_version": torch.version.cuda,
                },
            }
        )
        write_json(manifest_path, manifest, overwrite=True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_sequence_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=args.chat_template,
            mapping={
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            map_eos_token=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(train_path),
                "validation": str(validation_path),
            },
        )
        dataset = dataset.map(
            format_verifier_record,
            fn_kwargs={"tokenizer": tokenizer},
            load_from_cache_file=False,
        )
        training_arguments = SFTConfig(
            output_dir=str(output_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="epoch",
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=args.max_sequence_length,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            args=training_arguments,
        )
        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        manifest.update(
            {
                "status": "complete",
                "updated_utc": utc_now(),
                "finished_utc": utc_now(),
                "training_seconds": time.perf_counter() - started,
                "outputs": {
                    "adapter_config_sha256": first_existing_sha256(
                        output_dir, ("adapter_config.json",)
                    ),
                    "adapter_weights_sha256": first_existing_sha256(
                        output_dir,
                        ("adapter_model.safetensors", "adapter_model.bin"),
                    ),
                },
            }
        )
        write_json(manifest_path, manifest, overwrite=True)
        print(f"Verifier adapter saved to {output_dir}")
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "updated_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
            }
        )
        write_json(manifest_path, manifest, overwrite=True)
        raise


if __name__ == "__main__":
    main()

