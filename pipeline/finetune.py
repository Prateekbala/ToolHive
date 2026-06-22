"""
Phase 1 — LoRA fine-tuning.

Trains a domain-specific LoRA adapter using TRL SFTTrainer.
Prefers Unsloth (2-3x faster, lower VRAM); falls back to plain PEFT+TRL.

Design constraints:
  - Adapter rank is fixed (LORA_RANK=16) so the S-LoRA Unified Paging serving
    layer can co-serve all specialists without rank-heterogeneity tail-latency
    penalties. Never pass rank as a parameter.
  - Every run writes to a versioned candidate directory — never overwrites the
    active adapter. Promotion is the loop's responsibility.
  - Requires CUDA. Raises RuntimeError("training requires GPU") if unavailable
    so the loop can detect and skip training in CPU-only mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.runtime.schema import DomainSpec
    from pipeline.datagen import TrainingExample

# Fixed rank — matches goal.md and harness.py init_empty_adapter().
# Changing this requires retraining all existing specialists.
LORA_RANK: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: list[str] = ["q_proj", "v_proj"]


@dataclass
class TrainConfig:
    base_model: str
    output_dir: Path
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    seed: int = 42


def train(
    domain: "DomainSpec",
    examples: list["TrainingExample"],
    base_model: str,
    output_dir: str | Path,
    goal: dict[str, Any],
) -> Path:
    """
    Fine-tune a LoRA adapter on examples.

    Returns the Path to the candidate adapter directory.
    Raises RuntimeError("training requires GPU") if CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "training requires GPU — CUDA not available. "
            "Datagen and eval still work on CPU."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = Path(output_dir) / f"candidate_{timestamp}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        base_model=base_model,
        output_dir=candidate_dir,
        seed=goal.get("seed", 42),
        num_epochs=goal.get("num_epochs", 3),
    )

    success = _try_unsloth(config, examples)
    if not success:
        _train_peft_trl(config, examples)

    # Write a manifest alongside the adapter weights
    manifest = {
        "base_model": base_model,
        "domain": domain.domain,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "n_train_examples": len(examples),
        "trained_at": timestamp,
    }
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return candidate_dir


def _build_lora_config() -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )


def _format_for_sft(example: "TrainingExample", tokenizer: Any) -> str:
    """
    Convert a TrainingExample to a single string for SFTTrainer.

    Applies the tokenizer's chat template to the full conversation including
    the assistant's expected tool call. The assistant turn is always the
    compact JSON tool call with no extra whitespace or explanation.
    """
    assistant_content = json.dumps(
        example.expected_tool_call, separators=(",", ":")
    )
    full_messages = list(example.messages) + [
        {"role": "assistant", "content": assistant_content}
    ]
    return tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )


def _make_hf_dataset(examples: list["TrainingExample"], tokenizer: Any) -> Any:
    from datasets import Dataset

    rows = [{"text": _format_for_sft(ex, tokenizer)} for ex in examples]
    return Dataset.from_list(rows)


def _try_unsloth(config: TrainConfig, examples: list["TrainingExample"]) -> bool:
    """
    Attempt Unsloth-accelerated training.
    Returns True if training completed, False if Unsloth is unavailable.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        return False

    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES + ["k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    dataset = _make_hf_dataset(examples, tokenizer)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            seed=config.seed,
            save_strategy="no",
            logging_steps=10,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
        ),
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return True


def _train_peft_trl(config: TrainConfig, examples: list["TrainingExample"]) -> None:
    """Fallback: standard PEFT + TRL SFTTrainer (no Unsloth)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model
    from trl import SFTTrainer, SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, _build_lora_config())
    model.print_trainable_parameters()

    dataset = _make_hf_dataset(examples, tokenizer)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            seed=config.seed,
            save_strategy="no",
            logging_steps=10,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
        ),
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
