"""Prepare and validate clean, leakage-free train/validation/test JSONL splits.

Downloads `RenatoSpessotto/olist-aspectos-reviews-ptbr` (`silver` for train/validation,
`gold` for test), filters short texts (< 15 chars), deduplicates across splits
(test -> validation -> train priority) to prevent data leakage, maps rows
deterministically to the strict `TriageOutput` Pydantic schema, and performs
stratified sampling for `train` and `validation`.

Usage:
    python -m src.data.prepare_dataset --output-dir /content/dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.schemas import ACAO_LABELS, TriageOutput

_WS_RE = re.compile(r"\s+")

_MISSING_WRONG_KW = (
    "faltou",
    "faltando",
    "falta ",
    "errado",
    "errada",
    "trocado",
    "trocada",
    "incompleto",
    "incompleta",
    "veio outro",
    "não veio",
    "nao veio",
    "diferente do",
)
_PAYMENT_KW = (
    "cobrança",
    "cobranca",
    "cobrado",
    "estorno",
    "reembolso",
    "pagamento",
    "cartão",
    "cartao",
    "boleto",
    "pix",
    "fatura",
    "duplicad",
)
_DELAY_KW = (
    "atras",
    "demor",
    "não chegou",
    "nao chegou",
    "prazo",
    "extravio",
    "extraviad",
    "entregador",
    "correio",
    "dias esperando",
)
_SUPPORT_KW = (
    "atendimento",
    "suporte",
    "sac",
    "ninguém responde",
    "ninguem responde",
    "sem resposta",
    "contato",
    "chat",
    "vendedor",
    "mal educad",
)
_QUALITY_KW = (
    "quebrad",
    "danificad",
    "defeito",
    "estraga",
    "frio",
    "fria",
    "murch",
    "péssim",
    "pessim",
    "horrível",
    "horrivel",
    "qualidade",
    "rasgad",
    "amassad",
    "ruim",
)
_CRITICAL_KW = (
    "nunca chegou",
    "não recebi",
    "nao recebi",
    "procon",
    "absurdo",
    "golpe",
    "estorno",
    "reembolso",
)


def normalize_text(text: str) -> str:
    """Normalize text for exact/near-exact deduplication."""
    return _WS_RE.sub(" ", text.strip().lower())


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving original casing and punctuation."""
    return _WS_RE.sub(" ", str(text).strip())


def _truncate_words(text: str, max_words: int = 15) -> str:
    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def map_olist_row_to_triage(row: dict[str, Any]) -> TriageOutput:
    """Deterministically map an Olist aspect-review row to `TriageOutput`."""
    text = clean_text(row.get("text", ""))
    text_l = text.lower()
    motivo = clean_text(row.get("motivo_principal") or "")
    combined_l = f"{text_l} {motivo.lower()}".strip()

    score = int(row.get("review_score", 3))
    entrega = str(row.get("entrega", "ausente")).lower()
    produto = str(row.get("produto", "ausente")).lower()
    atendimento = str(row.get("atendimento", "ausente")).lower()
    preco = str(row.get("preco", "ausente")).lower()
    embalagem = str(row.get("embalagem", "ausente")).lower()

    has_neg_aspect = "negativo" in {entrega, produto, atendimento, preco, embalagem}

    # 1. Sentimento
    if score >= 4:
        sentimento = "positivo"
    elif score == 3:
        sentimento = "neutro"
    else:
        sentimento = "negativo"

    # 2. Categoria de incidente
    if sentimento == "positivo" and not has_neg_aspect:
        categoria = "elogio_geral"
    elif any(kw in combined_l for kw in _MISSING_WRONG_KW):
        categoria = "item_errado_ou_faltando"
    elif preco == "negativo" or any(kw in combined_l for kw in _PAYMENT_KW):
        categoria = "pagamento_cobranca"
    elif entrega == "negativo" or any(kw in combined_l for kw in _DELAY_KW):
        categoria = "entrega_atrasada"
    elif atendimento == "negativo" or any(kw in combined_l for kw in _SUPPORT_KW):
        categoria = "atendimento"
    elif produto == "negativo" or embalagem == "negativo" or any(
        kw in combined_l for kw in _QUALITY_KW
    ):
        categoria = "qualidade_produto"
    elif sentimento == "positivo":
        categoria = "elogio_geral"
    else:
        categoria = "outro"

    # 3. Severidade
    if sentimento == "positivo":
        severidade = "baixa"
    elif sentimento == "neutro":
        severidade = "media" if has_neg_aspect else "baixa"
    else:
        if score == 1 and any(kw in combined_l for kw in _CRITICAL_KW):
            severidade = "critica"
        elif score == 1:
            severidade = "alta"
        else:
            severidade = "media"

    # 4. Requer ação humana
    requer_acao_humana = severidade in {"media", "alta", "critica"} and categoria != "elogio_geral"

    # 5. Ação sugerida (consistent across silver & gold)
    if not requer_acao_humana or categoria == "elogio_geral":
        acao_sugerida = "nenhuma"
    elif severidade == "critica" or categoria in {"pagamento_cobranca", "atendimento"}:
        acao_sugerida = "escalar_suporte"
    elif categoria == "item_errado_ou_faltando" and severidade == "alta":
        acao_sugerida = "estorno_total"
    elif categoria in {"item_errado_ou_faltando", "qualidade_produto"}:
        acao_sugerida = "estorno_parcial"
    elif categoria == "entrega_atrasada":
        acao_sugerida = "acompanhar" if severidade == "media" else "escalar_suporte"
    else:
        acao_sugerida = "acompanhar"

    assert acao_sugerida in ACAO_LABELS

    # 6. Resumo curto (<= 15 palavras)
    if len(motivo) >= 5:
        resumo = _truncate_words(motivo, max_words=15)
        resumo = resumo[0].upper() + resumo[1:]
    else:
        resumo = _truncate_words(text, max_words=15)

    if len(resumo) < 3:
        resumo = f"Review classificado como {categoria}."

    return TriageOutput(
        sentimento=sentimento,
        categoria_incidente=categoria,
        severidade=severidade,
        requer_acao_humana=requer_acao_humana,
        acao_sugerida=acao_sugerida,
        resumo_curto=resumo,
    )


def deduplicate_splits(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    min_chars: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter short texts and deduplicate with strict priority: test > validation > train."""
    seen: set[str] = set()

    def _filter_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in rows:
            text = clean_text(row.get("text", ""))
            if len(text) < min_chars:
                continue
            norm = normalize_text(text)
            if norm in seen:
                continue
            seen.add(norm)
            kept.append({**row, "text": text})
        return kept

    clean_test = _filter_split(test_rows)
    clean_val = _filter_split(val_rows)
    clean_train = _filter_split(train_rows)
    return clean_train, clean_val, clean_test


def stratified_sample(
    records: list[dict[str, Any]],
    target_size: int,
    stratify_key: str = "categoria_incidente",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to `target_size` rows while balancing classes across `stratify_key`."""
    if target_size <= 0 or len(records) <= target_size:
        return records

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[str(rec[stratify_key])].append(rec)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    # Round-robin across buckets so minority categories are well represented
    while len(selected) < target_size:
        added_any = False
        for key in sorted(buckets.keys()):
            if buckets[key] and len(selected) < target_size:
                selected.append(buckets[key].pop())
                added_any = True
        if not added_any:
            break

    rng.shuffle(selected)
    return selected


def _convert_rows(rows: list[dict[str, Any]], split_prefix: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        triage = map_olist_row_to_triage(row)
        converted.append(
            {
                "id": str(row.get("review_id", f"{split_prefix}_{idx:05d}")),
                "text": row["text"],
                **triage.model_dump(),
            }
        )
    return converted


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_olist_dataset(
    output_dir: Path,
    train_size: int = 2000,
    val_size: int = 250,
    min_chars: int = 15,
    seed: int = 42,
) -> dict[str, int]:
    """Download, clean, deduplicate, map, and save train/validation/test JSONL files."""
    from datasets import load_dataset

    silver = load_dataset("RenatoSpessotto/olist-aspectos-reviews-ptbr", "silver")
    gold = load_dataset("RenatoSpessotto/olist-aspectos-reviews-ptbr", "gold")

    raw_train = [dict(r) for r in silver["train"]]
    raw_val = [dict(r) for r in silver["validation"]]
    raw_test = [dict(r) for r in gold["test"]]

    clean_train, clean_val, clean_test = deduplicate_splits(
        train_rows=raw_train,
        val_rows=raw_val,
        test_rows=raw_test,
        min_chars=min_chars,
    )

    mapped_train = _convert_rows(clean_train, "train")
    mapped_val = _convert_rows(clean_val, "val")
    mapped_test = _convert_rows(clean_test, "test")

    sampled_train = stratified_sample(mapped_train, target_size=train_size, seed=seed)
    sampled_val = stratified_sample(mapped_val, target_size=val_size, seed=seed)

    _write_jsonl(sampled_train, output_dir / "train.jsonl")
    _write_jsonl(sampled_val, output_dir / "validation.jsonl")
    _write_jsonl(mapped_test, output_dir / "test.jsonl")

    counts = {
        "train": len(sampled_train),
        "validation": len(sampled_val),
        "test": len(mapped_test),
    }
    print(f"Saved clean dataset to {output_dir}: {counts}")
    for split_name, rows in (
        ("train", sampled_train),
        ("validation", sampled_val),
        ("test", mapped_test),
    ):
        dist = dict(Counter(r["categoria_incidente"] for r in rows))
        print(f"  {split_name} categoria distribution: {dist}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean PT-BR delivery triage dataset")
    parser.add_argument("--output-dir", type=str, default="/content/dataset")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=250)
    parser.add_argument("--min-chars", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_olist_dataset(
        output_dir=Path(args.output_dir),
        train_size=args.train_size,
        val_size=args.val_size,
        min_chars=args.min_chars,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
