"""Deterministic Eval Harness for delivery-triage-llm-ptbr.

Measures:
- Strict JSON Validity (%) and Repaired JSON Validity (%)
- Strict Pydantic Schema Compliance (%)
- Macro-F1 for `categoria_incidente` and `sentimento` (penalizing invalid outputs)
- Accuracy for `severidade`, `acao_sugerida`, and `requer_acao_humana`
- Mean / p50 / p95 inference latency (ms)

Generates:
- `results/<run_name>_metrics.json`
- `results/<run_name>_predictions.jsonl`
- `results/confusion_matrix_<run_name>.png` (and `results/confusion_matrix.png`)
- `results/error_cases_<run_name>.jsonl` (top 20 worst errors)
- `results/benchmark_summary.md` (comparative table across all runs)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.prompts import SYSTEM_PROMPT, build_messages
from src.schemas import (
    CATEGORIA_LABELS,
    SENTIMENTO_LABELS,
    TriageOutput,
    parse_triage_output,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INVALID_LABEL = "__invalid__"

CANONICAL_RUN_ORDER = [
    "baseline_zero_shot",
    "baseline_few_shot",
    "qlora_v1",
    "qlora_v2",
]


@dataclass(frozen=True)
class EvalRecord:
    """Normalized ground-truth evaluation sample."""

    sample_id: str
    review_text: str
    ground_truth: TriageOutput


@dataclass(frozen=True)
class PredictionRecord:
    """Single model prediction with latency metadata."""

    sample_id: str
    review_text: str
    ground_truth: dict[str, Any]
    raw_output: str
    latency_ms: float


def set_deterministic_seed(seed: int = 42) -> None:
    """Fix RNG seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _strip_user_prefix(text: str) -> str:
    prefix = "Review do pedido:"
    cleaned = text.strip()
    if cleaned.startswith(prefix):
        return cleaned[len(prefix) :].strip()
    return cleaned


def _normalize_row(row: dict[str, Any], idx: int) -> EvalRecord:
    """Normalize a dataset row (ChatML messages, JSON target, or flat columns) to EvalRecord."""
    sample_id = str(row.get("id", row.get("sample_id", f"sample_{idx:04d}")))

    # Format 1: ChatML messages list
    if "messages" in row and isinstance(row["messages"], list):
        messages = row["messages"]
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
        if not user_msgs or not assistant_msgs:
            raise ValueError(f"Row {sample_id} has 'messages' but is missing user/assistant turns.")
        review_text = _strip_user_prefix(user_msgs[-1])
        gt_raw = assistant_msgs[-1]
        gt_dict = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        return EvalRecord(
            sample_id=sample_id,
            review_text=review_text,
            ground_truth=TriageOutput.model_validate(gt_dict),
        )

    # Extract review text from common column names
    review_text = None
    for col in ("review_text", "text", "review", "input", "comment", "review_comment_message"):
        if col in row and row[col] is not None:
            review_text = _strip_user_prefix(str(row[col]))
            break
    if not review_text:
        raise ValueError(f"Row {sample_id} missing review text column. Keys: {list(row.keys())}")

    # Format 2: JSON target column
    for target_col in ("target_json", "output", "label_json", "ground_truth", "target"):
        if target_col in row and row[target_col] is not None:
            raw_target = row[target_col]
            gt_dict = json.loads(raw_target) if isinstance(raw_target, str) else raw_target
            return EvalRecord(
                sample_id=sample_id,
                review_text=review_text,
                ground_truth=TriageOutput.model_validate(gt_dict),
            )

    # Format 3: Flat columns matching TriageOutput fields
    flat_fields = (
        "sentimento",
        "categoria_incidente",
        "severidade",
        "requer_acao_humana",
        "acao_sugerida",
        "resumo_curto",
    )
    if all(field in row for field in flat_fields):
        gt_dict = {field: row[field] for field in flat_fields}
        return EvalRecord(
            sample_id=sample_id,
            review_text=review_text,
            ground_truth=TriageOutput.model_validate(gt_dict),
        )

    raise ValueError(
        f"Row {sample_id} does not match supported schemas (messages, target_json, or flat fields)."
    )


def load_eval_records(
    dataset_path_or_id: str,
    split: str = "test",
    max_samples: int | None = None,
) -> list[EvalRecord]:
    """Load evaluation samples from local JSONL/Parquet/CSV or Hugging Face Hub."""
    path = Path(dataset_path_or_id)
    rows: list[dict[str, Any]] = []

    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                rows = data if isinstance(data, list) else data[split]
        elif suffix in {".parquet", ".csv"}:
            from datasets import load_dataset

            fmt = "parquet" if suffix == ".parquet" else "csv"
            ds = load_dataset(fmt, data_files=str(path), split="train")
            rows = [dict(r) for r in ds]
        else:
            raise ValueError(f"Unsupported local file extension: {suffix}")
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset_path_or_id, split=split)
        rows = [dict(r) for r in ds]

    if max_samples is not None:
        rows = rows[:max_samples]

    return [_normalize_row(row, idx) for idx, row in enumerate(rows)]


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str | None = None,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load Qwen2.5-3B-Instruct (and optional LoRA adapter) via Unsloth or Transformers."""
    # Prefer Unsloth on CUDA for ~2x faster inference on Colab T4
    try:
        from unsloth import FastLanguageModel

        model_to_load = adapter_path if adapter_path else base_model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_to_load,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if load_in_4bit and torch.cuda.is_available()
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_live_inference(
    records: list[EvalRecord],
    model: Any,
    tokenizer: Any,
    prompt_mode: str = "zero_shot",
    max_new_tokens: int = 256,
) -> list[PredictionRecord]:
    """Run deterministic greedy decoding (`do_sample=False`) on all eval records."""
    import torch

    device = next(model.parameters()).device
    predictions: list[PredictionRecord] = []

    for record in tqdm(records, desc=f"Evaluating ({prompt_mode})"):
        messages = build_messages(
            review_text=record.review_text,
            mode=prompt_mode,
            system_prompt=SYSTEM_PROMPT,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        generated_ids = outputs[0][input_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append(
            PredictionRecord(
                sample_id=record.sample_id,
                review_text=record.review_text,
                ground_truth=record.ground_truth.model_dump(),
                raw_output=raw_output,
                latency_ms=round(latency_ms, 2),
            )
        )

    return predictions


def _severity_error_weight(pred_error: dict[str, Any]) -> int:
    """Rank error cases so the 20 most informative failures surface first."""
    score = 0
    if not pred_error["json_valid"]:
        score += 100
    elif not pred_error["schema_valid"]:
        score += 80

    gt = pred_error["ground_truth"]
    pred = pred_error["predicted"] or {}

    if gt.get("severidade") in {"critica", "alta"} and pred.get("severidade") != gt.get(
        "severidade"
    ):
        score += 40
    if pred.get("categoria_incidente") != gt.get("categoria_incidente"):
        score += 30
    if pred.get("sentimento") != gt.get("sentimento"):
        score += 20
    if pred.get("acao_sugerida") != gt.get("acao_sugerida"):
        score += 10
    if pred.get("requer_acao_humana") != gt.get("requer_acao_humana"):
        score += 10
    return score


def compute_metrics(
    predictions: list[PredictionRecord],
    run_name: str,
    model_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    """Compute all deterministic metrics, error cases, and category label vectors."""
    if not predictions:
        raise ValueError("Cannot compute metrics on an empty prediction list.")

    n = len(predictions)
    json_valid_count = 0
    json_repaired_valid_count = 0
    schema_valid_count = 0
    schema_repaired_valid_count = 0

    y_true_cat: list[str] = []
    y_pred_cat: list[str] = []
    y_true_sent: list[str] = []
    y_pred_sent: list[str] = []
    y_true_sev: list[str] = []
    y_pred_sev: list[str] = []
    y_true_acao: list[str] = []
    y_pred_acao: list[str] = []
    y_true_human: list[int] = []
    y_pred_human: list[int] = []
    latencies: list[float] = []

    error_cases: list[dict[str, Any]] = []

    for pred in predictions:
        gt = pred.ground_truth
        latencies.append(pred.latency_ms)

        strict_res = parse_triage_output(pred.raw_output, allow_repair=False)
        repaired_res = (
            strict_res
            if strict_res.schema_valid
            else parse_triage_output(pred.raw_output, allow_repair=True)
        )

        if strict_res.json_valid:
            json_valid_count += 1
        if strict_res.json_valid or (
            repaired_res.parsed_dict is not None and repaired_res.json_repaired
        ):
            json_repaired_valid_count += 1
        if strict_res.schema_valid:
            schema_valid_count += 1
        if repaired_res.schema_valid:
            schema_repaired_valid_count += 1

        y_true_cat.append(gt["categoria_incidente"])
        y_true_sent.append(gt["sentimento"])
        y_true_sev.append(gt["severidade"])
        y_true_acao.append(gt["acao_sugerida"])
        y_true_human.append(int(bool(gt["requer_acao_humana"])))

        if strict_res.validated is not None:
            v = strict_res.validated
            y_pred_cat.append(v.categoria_incidente)
            y_pred_sent.append(v.sentimento)
            y_pred_sev.append(v.severidade)
            y_pred_acao.append(v.acao_sugerida)
            y_pred_human.append(int(v.requer_acao_humana))
            pred_dict: dict[str, Any] | None = v.model_dump()
        else:
            y_pred_cat.append(INVALID_LABEL)
            y_pred_sent.append(INVALID_LABEL)
            y_pred_sev.append(INVALID_LABEL)
            y_pred_acao.append(INVALID_LABEL)
            y_pred_human.append(-1)
            pred_dict = strict_res.parsed_dict

        has_error = (
            not strict_res.schema_valid
            or pred_dict is None
            or pred_dict.get("categoria_incidente") != gt["categoria_incidente"]
            or pred_dict.get("sentimento") != gt["sentimento"]
            or pred_dict.get("severidade") != gt["severidade"]
            or pred_dict.get("acao_sugerida") != gt["acao_sugerida"]
            or pred_dict.get("requer_acao_humana") != gt["requer_acao_humana"]
        )
        if has_error:
            err_entry = {
                "sample_id": pred.sample_id,
                "review_text": pred.review_text,
                "ground_truth": gt,
                "predicted": pred_dict,
                "raw_output": pred.raw_output,
                "json_valid": strict_res.json_valid,
                "schema_valid": strict_res.schema_valid,
                "parse_error": strict_res.error,
                "latency_ms": pred.latency_ms,
            }
            err_entry["error_priority"] = _severity_error_weight(err_entry)
            error_cases.append(err_entry)

    # Sort error cases descending by severity/impact
    error_cases.sort(key=lambda x: x["error_priority"], reverse=True)

    # Macro-F1 evaluated strictly over the true label set (invalid outputs count as FN)
    present_cat_labels = [lbl for lbl in CATEGORIA_LABELS if lbl in set(y_true_cat)]
    present_sent_labels = [lbl for lbl in SENTIMENTO_LABELS if lbl in set(y_true_sent)]

    macro_f1_cat = float(
        f1_score(
            y_true_cat,
            y_pred_cat,
            labels=present_cat_labels,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1_cat = float(
        f1_score(
            y_true_cat,
            y_pred_cat,
            labels=present_cat_labels,
            average="weighted",
            zero_division=0,
        )
    )
    macro_f1_sent = float(
        f1_score(
            y_true_sent,
            y_pred_sent,
            labels=present_sent_labels,
            average="macro",
            zero_division=0,
        )
    )

    acc_sev = float(accuracy_score(y_true_sev, y_pred_sev))
    acc_acao = float(accuracy_score(y_true_acao, y_pred_acao))
    acc_human = float(accuracy_score(y_true_human, y_pred_human))

    latency_arr = np.array(latencies, dtype=float)
    latency_mean = float(np.mean(latency_arr))
    latency_p50 = float(np.percentile(latency_arr, 50))
    latency_p95 = float(np.percentile(latency_arr, 95))

    cat_report = classification_report(
        y_true_cat,
        y_pred_cat,
        labels=present_cat_labels,
        output_dict=True,
        zero_division=0,
    )
    sent_report = classification_report(
        y_true_sent,
        y_pred_sent,
        labels=present_sent_labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "run_name": run_name,
        "model_label": model_label,
        "total_samples": n,
        "json_validity_pct": round((json_valid_count / n) * 100.0, 2),
        "json_validity_with_repair_pct": round((json_repaired_valid_count / n) * 100.0, 2),
        "schema_compliance_pct": round((schema_valid_count / n) * 100.0, 2),
        "schema_compliance_with_repair_pct": round((schema_repaired_valid_count / n) * 100.0, 2),
        "macro_f1_categoria": round(macro_f1_cat, 4),
        "weighted_f1_categoria": round(weighted_f1_cat, 4),
        "macro_f1_sentimento": round(macro_f1_sent, 4),
        "accuracy_severidade": round(acc_sev, 4),
        "accuracy_acao_sugerida": round(acc_acao, 4),
        "accuracy_requer_acao_humana": round(acc_human, 4),
        "latency_mean_ms": round(latency_mean, 2),
        "latency_p50_ms": round(latency_p50, 2),
        "latency_p95_ms": round(latency_p95, 2),
        "num_error_cases": len(error_cases),
        "classification_report_categoria": cat_report,
        "classification_report_sentimento": sent_report,
    }

    return metrics, error_cases, y_true_cat, y_pred_cat


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot and save a readable confusion matrix PNG for `categoria_incidente`."""
    labels = list(CATEGORIA_LABELS)
    if INVALID_LABEL in y_pred:
        labels.append(INVALID_LABEL)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Ground Truth (categoria_incidente)",
        xlabel="Predicted (categoria_incidente)",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def update_baseline_metrics_bundle(results_dir: Path, metrics: dict[str, Any]) -> None:
    """Maintain consolidated `results/baseline_metrics.json` for Zero-Shot & Few-Shot runs."""
    run_name = metrics["run_name"]
    if run_name not in {"baseline_zero_shot", "baseline_few_shot"}:
        return

    bundle_path = results_dir / "baseline_metrics.json"
    bundle: dict[str, Any] = {}
    if bundle_path.exists():
        with bundle_path.open("r", encoding="utf-8") as f:
            bundle = json.load(f)

    key = "zero_shot" if run_name == "baseline_zero_shot" else "few_shot"
    bundle[key] = {
        k: v
        for k, v in metrics.items()
        if not k.startswith("classification_report_")
    }
    with bundle_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)


def build_benchmark_summary(results_dir: Path) -> Path:
    """Aggregate all `*_metrics.json` runs into `results/benchmark_summary.md`."""
    metric_files = sorted(results_dir.glob("*_metrics.json"))
    runs: list[dict[str, Any]] = []
    for path in metric_files:
        if path.name == "baseline_metrics.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "run_name" in data and "macro_f1_categoria" in data:
            runs.append(data)

    def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
        name = item["run_name"]
        idx = CANONICAL_RUN_ORDER.index(name) if name in CANONICAL_RUN_ORDER else 99
        return (idx, name)

    runs.sort(key=_sort_key)

    lines = [
        "# Benchmark Summary — `delivery-triage-llm-ptbr`",
        "",
        "Avaliação determinística (`do_sample=False`, `seed=42`) no conjunto `test`.",
        "",
        (
            "| Modelo / Configuração | JSON Valid (%) | Schema Valid (%) | "
            "Macro-F1 (Categoria) | Macro-F1 (Sentimento) | Acc (Severidade) | "
            "Acc (Ação) | Latência p50 / p95 (ms) |"
        ),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in runs:
        lines.append(
            f"| {r['model_label']} "
            f"| {r['json_validity_pct']:.1f}% "
            f"| {r['schema_compliance_pct']:.1f}% "
            f"| {r['macro_f1_categoria']:.4f} "
            f"| {r['macro_f1_sentimento']:.4f} "
            f"| {r['accuracy_severidade'] * 100:.1f}% "
            f"| {r['accuracy_acao_sugerida'] * 100:.1f}% "
            f"| {r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} |"
        )

    lines.append("")
    summary_path = results_dir / "benchmark_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def save_eval_artifacts(
    predictions: list[PredictionRecord],
    metrics: dict[str, Any],
    error_cases: list[dict[str, Any]],
    y_true_cat: list[str],
    y_pred_cat: list[str],
    results_dir: Path,
    top_k_errors: int = 20,
) -> None:
    """Persist metrics, predictions, confusion matrix, error cases, and benchmark table."""
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = metrics["run_name"]

    metrics_path = results_dir / f"{run_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    preds_path = results_dir / f"{run_name}_predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(
                json.dumps(
                    {
                        "sample_id": p.sample_id,
                        "review_text": p.review_text,
                        "ground_truth": p.ground_truth,
                        "raw_output": p.raw_output,
                        "latency_ms": p.latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    errors_path = results_dir / f"error_cases_{run_name}.jsonl"
    with errors_path.open("w", encoding="utf-8") as f:
        for err in error_cases[:top_k_errors]:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
    shutil.copyfile(errors_path, results_dir / "error_cases.jsonl")

    cm_path = results_dir / f"confusion_matrix_{run_name}.png"
    save_confusion_matrix(
        y_true=y_true_cat,
        y_pred=y_pred_cat,
        output_path=cm_path,
        title=f"Confusion Matrix — {metrics['model_label']}",
    )
    shutil.copyfile(cm_path, results_dir / "confusion_matrix.png")

    update_baseline_metrics_bundle(results_dir, metrics)
    build_benchmark_summary(results_dir)


def load_predictions_jsonl(path: Path) -> list[PredictionRecord]:
    """Load saved predictions for offline metric recomputation."""
    records: list[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                PredictionRecord(
                    sample_id=str(row["sample_id"]),
                    review_text=str(row["review_text"]),
                    ground_truth=row["ground_truth"],
                    raw_output=str(row["raw_output"]),
                    latency_ms=float(row.get("latency_ms", 0.0)),
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Eval Harness for Delivery Triage")
    parser.add_argument("--dataset", type=str, help="Local path (.jsonl/.parquet/.csv) or HF repo")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="Base model HF ID or local path",
    )
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
    )
    parser.add_argument("--run-name", type=str, default="baseline_zero_shot")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=None,
        help="Evaluate pre-generated predictions JSONL offline without loading a model",
    )
    parser.add_argument(
        "--build-summary-only",
        action="store_true",
        help="Only rebuild results/benchmark_summary.md from existing *_metrics.json files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if args.build_summary_only:
        summary_path = build_benchmark_summary(results_dir)
        print(f"Benchmark summary updated: {summary_path}")
        return

    set_deterministic_seed(args.seed)

    model_label = args.model_label or args.run_name
    if args.predictions_file:
        predictions = load_predictions_jsonl(Path(args.predictions_file))
    else:
        if not args.dataset:
            raise SystemExit("Error: --dataset is required unless --predictions-file is provided.")
        records = load_eval_records(
            dataset_path_or_id=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
        )
        model, tokenizer = load_model_and_tokenizer(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
        )
        predictions = run_live_inference(
            records=records,
            model=model,
            tokenizer=tokenizer,
            prompt_mode=args.prompt_mode,
            max_new_tokens=args.max_new_tokens,
        )

    metrics, error_cases, y_true_cat, y_pred_cat = compute_metrics(
        predictions=predictions,
        run_name=args.run_name,
        model_label=model_label,
    )
    save_eval_artifacts(
        predictions=predictions,
        metrics=metrics,
        error_cases=error_cases,
        y_true_cat=y_true_cat,
        y_pred_cat=y_pred_cat,
        results_dir=results_dir,
    )

    print("\n" + "=" * 64)
    print(f"Run: {metrics['model_label']} ({metrics['total_samples']} samples)")
    print(f"  JSON Validity (strict): {metrics['json_validity_pct']:.2f}%")
    print(f"  Schema Compliance:      {metrics['schema_compliance_pct']:.2f}%")
    print(f"  Macro-F1 (Categoria):   {metrics['macro_f1_categoria']:.4f}")
    print(f"  Macro-F1 (Sentimento):  {metrics['macro_f1_sentimento']:.4f}")
    print(f"  Accuracy (Severidade):  {metrics['accuracy_severidade'] * 100:.2f}%")
    print(f"  Accuracy (Ação):        {metrics['accuracy_acao_sugerida'] * 100:.2f}%")
    print(
        f"  Latency mean/p50/p95:   {metrics['latency_mean_ms']:.1f} / "
        f"{metrics['latency_p50_ms']:.1f} / {metrics['latency_p95_ms']:.1f} ms"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
