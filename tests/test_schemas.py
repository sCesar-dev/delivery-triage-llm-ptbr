"""Unit tests for strict Pydantic TriageOutput contract and JSON parser."""

from __future__ import annotations

import json

from src.schemas import TriageOutput, parse_triage_output

VALID_PAYLOAD = {
    "sentimento": "negativo",
    "categoria_incidente": "entrega_atrasada",
    "severidade": "alta",
    "requer_acao_humana": True,
    "acao_sugerida": "escalar_suporte",
    "resumo_curto": "Pedido atrasou mais de uma hora e chegou frio.",
}


def test_parse_strict_valid_json() -> None:
    raw = json.dumps(VALID_PAYLOAD, ensure_ascii=False)
    res = parse_triage_output(raw, allow_repair=False)
    assert res.json_valid is True
    assert res.schema_valid is True
    assert res.json_repaired is False
    assert isinstance(res.validated, TriageOutput)
    assert res.validated.categoria_incidente == "entrega_atrasada"


def test_parse_markdown_fenced_json_fails_strict_passes_repair() -> None:
    raw = f"Aqui está o JSON:\n```json\n{json.dumps(VALID_PAYLOAD, ensure_ascii=False)}\n```"

    strict_res = parse_triage_output(raw, allow_repair=False)
    assert strict_res.json_valid is False
    assert strict_res.schema_valid is False

    repaired_res = parse_triage_output(raw, allow_repair=True)
    assert repaired_res.json_valid is False
    assert repaired_res.json_repaired is True
    assert repaired_res.schema_valid is True


def test_parse_rejects_invalid_enum_and_extra_fields() -> None:
    bad_enum = {**VALID_PAYLOAD, "categoria_incidente": "atraso_motoboy"}
    res1 = parse_triage_output(json.dumps(bad_enum))
    assert res1.json_valid is True
    assert res1.schema_valid is False
    assert res1.error is not None and "schema_validation_error" in res1.error

    extra_field = {**VALID_PAYLOAD, "campo_inventado": 123}
    res2 = parse_triage_output(json.dumps(extra_field))
    assert res2.json_valid is True
    assert res2.schema_valid is False


def test_parse_rejects_string_boolean_for_requer_acao_humana() -> None:
    bad_bool = {**VALID_PAYLOAD, "requer_acao_humana": "true"}
    res = parse_triage_output(json.dumps(bad_bool))
    assert res.json_valid is True
    assert res.schema_valid is False
