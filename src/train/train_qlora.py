"""QLoRA 4-bit fine-tuning CLI using Unsloth + TRL SFTTrainer + train_on_responses_only.

Usage (on Google Colab T4 / Kaggle T4):
    python -m src.train.train_qlora \
        --config configs/train_qlora_v1.yaml \
        --dataset sCesar-dev/delivery-triage-reviews-ptbr
"""

from __future__ import annotations

try:  # Unsloth must be imported before trl/transformers/peft for full optimizations
    import unsloth  # noqa: F401  # isort: skip
except ImportError:
    pass

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib
import yaml

from src.prompts import SYSTEM_PROMPT, build_messages
from src.schemas import TriageOutput

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Convert any supported dataset row format into ChatML `messages`."""
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]

    review_text = None
    for col in ("review_text", "text", "review", "input", "comment", "review_comment_message"):
        if col in row and row[col] is not None:
            review_text = str(row[col])
            break
    if not review_text:
        raise ValueError(f"Could not find review text column in keys: {list(row.keys())}")

    target_dict: dict[str, Any] | None = None
    for col in ("target_json", "output", "label_json", "ground_truth", "target"):
        if col in row and row[col] is not None:
            raw = row[col]
            target_dict = json.loads(raw) if isinstance(raw, str) else raw
            break

    if target_dict is None:
        flat_fields = (
            "sentimento",
            "categoria_incidente",
            "severidade",
            "requer_acao_humana",
            "acao_sugerida",
            "resumo_curto",
        )
        if all(f in row for f in flat_fields):
            target_dict = {f: row[f] for f in flat_fields}

    if target_dict is None:
        raise ValueError(f"Could not find target JSON or flat schema fields in {list(row.keys())}")

    # Validate target against Pydantic schema before training
    validated = TriageOutput.model_validate(target_dict)
    return build_messages(
        review_text=review_text,
        mode="zero_shot",
        system_prompt=SYSTEM_PROMPT,
        target_json=validated.model_dump(),
    )


def load_train_val_datasets(
    dataset_path_or_id: str,
    train_split: str = "train",
    val_split: str = "validation",
):
    """Load train and validation splits from HF Hub or a local directory of JSONL/Parquet files."""
    from datasets import Dataset, load_dataset

    path = Path(dataset_path_or_id)
    if path.is_dir():
        splits = {}
        for split_name, candidates in (
            ("train", [f"{train_split}.jsonl", f"{train_split}.parquet"]),
            ("val", [f"{val_split}.jsonl", "val.jsonl", f"{val_split}.parquet", "val.parquet"]),
        ):
            found = next((path / c for c in candidates if (path / c).exists()), None)
            if found is None:
                raise FileNotFoundError(f"Missing {split_name} file in {path}. Tried: {candidates}")
            fmt = "json" if found.suffix == ".jsonl" else "parquet"
            splits[split_name] = load_dataset(fmt, data_files=str(found), split="train")
        return splits["train"], splits["val"]

    ds_dict = load_dataset(dataset_path_or_id)
    train_ds: Dataset = ds_dict[train_split]
    val_key = val_split if val_split in ds_dict else "val"
    val_ds: Dataset = ds_dict[val_key]
    return train_ds, val_ds


def save_loss_curve(log_history: list[dict[str, Any]], output_dir: Path, run_name: str) -> None:
    """Save training vs validation loss history JSON and plot PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{run_name}_training_history.json"
    history_path.write_text(json.dumps(log_history, indent=2), encoding="utf-8")

    train_steps = [x["step"] for x in log_history if "loss" in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

    if not train_steps:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_steps, train_losses, label="train_loss", linewidth=1.8)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="eval_loss", marker="o", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss (response tokens only)")
    ax.set_title(f"Training vs Eval Loss — {run_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_loss_curve.png", dpi=160)
    plt.close(fig)


def train(
    config: dict[str, Any],
    dataset_path_or_id: str,
    train_split: str = "train",
    val_split: str = "validation",
    results_dir: Path = Path("results"),
) -> Path:
    """Run QLoRA 4-bit SFT with Unsloth and return the saved adapter directory."""
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    seed = int(config.get("seed", 42))
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        dtype=model_cfg.get("dtype"),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=seed,
    )

    train_ds, val_ds = load_train_val_datasets(
        dataset_path_or_id=dataset_path_or_id,
        train_split=train_split,
        val_split=val_split,
    )

    def _format_to_text(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        size = len(next(iter(batch.values())))
        texts: list[str] = []
        for i in range(size):
            row = {k: batch[k][i] for k in batch}
            messages = _row_to_messages(row)
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(formatted)
        return {"text": texts}

    train_ds = train_ds.map(_format_to_text, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_format_to_text, batched=True, remove_columns=val_ds.column_names)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    # TRL >= 0.20 renamed `max_seq_length` -> `max_length`
    seq_len_key = "max_length" if "max_length" in sft_params else "max_seq_length"
    sft_kwargs: dict[str, Any] = dict(
        output_dir=train_cfg["output_dir"],
        dataset_text_field="text",
        **{seq_len_key: int(model_cfg["max_seq_length"])},
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optim=str(train_cfg["optim"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy=str(train_cfg["eval_strategy"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_strategy=str(train_cfg["save_strategy"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=str(train_cfg["metric_for_best_model"]),
        greater_is_better=bool(train_cfg["greater_is_better"]),
        fp16=not use_bf16,
        bf16=use_bf16,
        seed=seed,
        report_to="none",
        packing=False,
    )
    sft_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sft_params})

    # TRL >= 0.12 renamed `tokenizer` -> `processing_class`
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
        **{tok_key: tokenizer},
    )

    if train_cfg.get("train_on_responses_only", True):
        trainer = train_on_responses_only(
            trainer,
            instruction_part=train_cfg.get("instruction_part", "<|im_start|>user\n"),
            response_part=train_cfg.get("response_part", "<|im_start|>assistant\n"),
        )

    trainer.train()

    adapter_dir = Path(train_cfg["adapter_save_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    save_loss_curve(trainer.state.log_history, results_dir, config["run_name"])
    print(f"Training complete. Best LoRA adapter saved to: {adapter_dir}")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QLoRA adapter with Unsloth")
    parser.add_argument("--config", type=str, default="configs/train_qlora_v1.yaml")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset ID or local dir")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train(
        config=cfg,
        dataset_path_or_id=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
