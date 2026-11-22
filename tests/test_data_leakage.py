"""Unit tests for dataset preparation, schema mapping, and zero data leakage."""

from __future__ import annotations

from src.data.prepare_dataset import (
    deduplicate_splits,
    map_olist_row_to_triage,
    normalize_text,
    stratified_sample,
)


def test_deduplicate_splits_prevents_leakage_with_test_priority() -> None:
    train = [
        {"text": "Pedido atrasou muito e chegou frio!"},
        {"text": "Curto"},  # < 15 chars -> filtered
        {"text": "Excelente produto, chegou antes do prazo."},
    ]
    val = [
        {"text": "  PEDIDO   ATRASOU MUITO E CHEGOU FRIO! "},  # duplicate of test/train
        {"text": "Produto com defeito na embalagem."},
    ]
    test = [
        {"text": "Pedido atrasou muito e chegou frio!"},
    ]

    clean_train, clean_val, clean_test = deduplicate_splits(train, val, test, min_chars=15)

    train_norms = {normalize_text(r["text"]) for r in clean_train}
    val_norms = {normalize_text(r["text"]) for r in clean_val}
    test_norms = {normalize_text(r["text"]) for r in clean_test}

    assert len(clean_test) == 1
    assert train_norms.isdisjoint(test_norms)
    assert val_norms.isdisjoint(test_norms)
    assert train_norms.isdisjoint(val_norms)


def test_map_olist_row_to_triage_validates_schema() -> None:
    neg_row = {
        "text": "Pedi dois itens e veio faltando um, absurdo nunca mais compro.",
        "review_score": 1,
        "entrega": "ausente",
        "produto": "negativo",
        "atendimento": "ausente",
        "preco": "ausente",
        "embalagem": "ausente",
    }
    res = map_olist_row_to_triage(neg_row)
    assert res.sentimento == "negativo"
    assert res.categoria_incidente == "item_errado_ou_faltando"
    assert res.severidade == "critica"
    assert res.requer_acao_humana is True
    assert res.acao_sugerida == "escalar_suporte"

    pos_row = {
        "text": "Chegou antes do prazo e muito bem embalado!",
        "review_score": 5,
        "entrega": "positivo",
        "produto": "positivo",
        "atendimento": "ausente",
        "preco": "ausente",
        "embalagem": "positivo",
    }
    res_pos = map_olist_row_to_triage(pos_row)
    assert res_pos.sentimento == "positivo"
    assert res_pos.categoria_incidente == "elogio_geral"
    assert res_pos.requer_acao_humana is False
    assert res_pos.acao_sugerida == "nenhuma"


def test_stratified_sample_balances_classes() -> None:
    records = [{"categoria_incidente": "elogio_geral", "id": i} for i in range(50)] + [
        {"categoria_incidente": "pagamento_cobranca", "id": 100 + i} for i in range(10)
    ]
    sampled = stratified_sample(records, target_size=16, seed=42)
    assert len(sampled) == 16
    cats = [r["categoria_incidente"] for r in sampled]
    assert cats.count("pagamento_cobranca") == 8
    assert cats.count("elogio_geral") == 8
