"""Unit tests for the deterministic Eval Harness."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.error_analysis import summarize_errors
from src.eval.evaluate import (
    PredictionRecord,
    build_benchmark_summary,
    compute_metrics,
    load_eval_records,
    save_eval_artifacts,
)


def _sample_gt(cat: str, sent: str, sev: str, acao: str, human: bool) -> dict:
    return {
        "sentimento": sent,
        "categoria_incidente": cat,
        "severidade": sev,
        "requer_acao_humana": human,
        "acao_sugerida": acao,
        "resumo_curto": f"Resumo curto de teste para {cat}.",
    }


def test_compute_metrics_penalizes_invalid_outputs() -> None:
    gt1 = _sample_gt("entrega_atrasada", "negativo", "alta", "escalar_suporte", True)
    gt2 = _sample_gt("elogio_geral", "positivo", "baixa", "nenhuma", False)
    gt3 = _sample_gt("item_errado_ou_faltando", "negativo", "media", "estorno_parcial", True)

    preds = [
        PredictionRecord("1", "Atrasou muito", gt1, json.dumps(gt1, ensure_ascii=False), 120.0),
        PredictionRecord("2", "Excelente!", gt2, json.dumps(gt2, ensure_ascii=False), 100.0),
        PredictionRecord("3", "Faltou refri", gt3, "Desculpe, não consegui gerar JSON", 200.0),
    ]

    metrics, errors, y_true, y_pred = compute_metrics(
        predictions=preds,
        run_name="baseline_zero_shot",
        model_label="Qwen2.5-3B Base (Zero-Shot)",
    )

    assert metrics["total_samples"] == 3
    assert metrics["json_validity_pct"] == 66.67
    assert metrics["schema_compliance_pct"] == 66.67
    # 2 classes perfect (F1=1.0), 1 class invalid (F1=0.0) -> Macro-F1 = 2/3 = 0.6667
    assert metrics["macro_f1_categoria"] == 0.6667
    assert metrics["accuracy_severidade"] == 0.6667
    assert metrics["accuracy_acao_sugerida"] == 0.6667
    assert metrics["latency_p50_ms"] == 120.0
    assert len(errors) == 1
    assert errors[0]["sample_id"] == "3"
    assert y_true == ["entrega_atrasada", "elogio_geral", "item_errado_ou_faltando"]
    assert y_pred[2] == "__invalid__"


def test_save_artifacts_and_benchmark_summary(tmp_path: Path) -> None:
    gt1 = _sample_gt("entrega_atrasada", "negativo", "alta", "escalar_suporte", True)
    gt2 = _sample_gt("elogio_geral", "positivo", "baixa", "nenhuma", False)

    # Zero-shot: 1 right, 1 broken JSON
    zs_preds = [
        PredictionRecord("1", "Atrasou", gt1, json.dumps(gt1, ensure_ascii=False), 150.0),
        PredictionRecord("2", "Ótimo", gt2, "```json\n{}\n```", 180.0),
    ]
    zs_metrics, zs_errors, yt, yp = compute_metrics(
        zs_preds, "baseline_zero_shot", "Qwen2.5-3B Base (Zero-Shot)"
    )
    save_eval_artifacts(zs_preds, zs_metrics, zs_errors, yt, yp, tmp_path)

    # QLoRA v1: 2 right
    v1_preds = [
        PredictionRecord("1", "Atrasou", gt1, json.dumps(gt1, ensure_ascii=False), 110.0),
        PredictionRecord("2", "Ótimo", gt2, json.dumps(gt2, ensure_ascii=False), 105.0),
    ]
    v1_metrics, v1_errors, yt2, yp2 = compute_metrics(
        v1_preds, "qlora_v1", "Qwen2.5-3B + QLoRA v1"
    )
    save_eval_artifacts(v1_preds, v1_metrics, v1_errors, yt2, yp2, tmp_path)

    assert (tmp_path / "baseline_metrics.json").exists()
    assert (tmp_path / "confusion_matrix_qlora_v1.png").exists()
    assert (tmp_path / "error_cases_baseline_zero_shot.jsonl").exists()

    summary_path = build_benchmark_summary(tmp_path)
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Qwen2.5-3B Base (Zero-Shot)" in summary_text
    assert "Qwen2.5-3B + QLoRA v1" in summary_text
    assert summary_text.index("Zero-Shot") < summary_text.index("QLoRA v1")

    err_summary = summarize_errors(tmp_path / "error_cases_baseline_zero_shot.jsonl")
    assert "invalid_json" in err_summary


def test_load_eval_records_supports_messages_and_flat_formats(tmp_path: Path) -> None:
    gt = _sample_gt("qualidade_produto", "negativo", "media", "estorno_parcial", True)

    # ChatML messages JSONL
    msg_file = tmp_path / "messages.jsonl"
    msg_row = {
        "id": "m1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Review do pedido:\nComida veio fria e murcha."},
            {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)},
        ],
    }
    msg_file.write_text(json.dumps(msg_row, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded_msg = load_eval_records(str(msg_file))
    assert len(loaded_msg) == 1
    assert loaded_msg[0].review_text == "Comida veio fria e murcha."
    assert loaded_msg[0].ground_truth.categoria_incidente == "qualidade_produto"

    # Flat columns JSONL
    flat_file = tmp_path / "flat.jsonl"
    flat_row = {"id": "f1", "text": "Comida fria.", **gt}
    flat_file.write_text(json.dumps(flat_row, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded_flat = load_eval_records(str(flat_file))
    assert len(loaded_flat) == 1
    assert loaded_flat[0].sample_id == "f1"
