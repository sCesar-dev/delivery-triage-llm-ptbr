"""Strict Pydantic v2 contract and deterministic JSON parser for delivery incident triage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Sentimento = Literal["positivo", "neutro", "negativo"]
CategoriaIncidente = Literal[
    "entrega_atrasada",
    "item_errado_ou_faltando",
    "qualidade_produto",
    "atendimento",
    "pagamento_cobranca",
    "elogio_geral",
    "outro",
]
Severidade = Literal["baixa", "media", "alta", "critica"]
AcaoSugerida = Literal[
    "nenhuma",
    "acompanhar",
    "estorno_parcial",
    "estorno_total",
    "escalar_suporte",
]

SENTIMENTO_LABELS: tuple[str, ...] = ("positivo", "neutro", "negativo")
CATEGORIA_LABELS: tuple[str, ...] = (
    "entrega_atrasada",
    "item_errado_ou_faltando",
    "qualidade_produto",
    "atendimento",
    "pagamento_cobranca",
    "elogio_geral",
    "outro",
)
SEVERIDADE_LABELS: tuple[str, ...] = ("baixa", "media", "alta", "critica")
ACAO_LABELS: tuple[str, ...] = (
    "nenhuma",
    "acompanhar",
    "estorno_parcial",
    "estorno_total",
    "escalar_suporte",
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class TriageOutput(BaseModel):
    """Strict output schema for PT-BR delivery review/incident triage."""

    model_config = ConfigDict(extra="forbid", strict=False)

    sentimento: Sentimento
    categoria_incidente: CategoriaIncidente
    severidade: Severidade
    requer_acao_humana: bool
    acao_sugerida: AcaoSugerida
    resumo_curto: str = Field(..., min_length=3, max_length=240)

    @field_validator("requer_acao_humana", mode="before")
    @classmethod
    def _reject_non_bool_strings(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        raise ValueError("requer_acao_humana must be a JSON boolean (true/false)")

    @field_validator("resumo_curto")
    @classmethod
    def _validate_resumo_curto(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("resumo_curto cannot be empty")
        return cleaned


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing and validating a raw LLM output string."""

    raw_text: str
    json_valid: bool
    json_repaired: bool
    schema_valid: bool
    parsed_dict: dict | None
    validated: TriageOutput | None
    error: str | None = None


def extract_json_candidate(raw_text: str) -> tuple[str | None, bool]:
    """Extract a JSON object from text if wrapped in markdown fences or surrounding prose.

    Returns:
        Tuple of (candidate_json_str, required_repair).
    """
    stripped = raw_text.strip()
    if not stripped:
        return None, False

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False

    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip(), True

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip(), True

    return None, False


def parse_triage_output(raw_text: str, allow_repair: bool = False) -> ParseResult:
    """Parse raw model output into a validated TriageOutput.

    By default (`allow_repair=False`), only pure JSON without markdown fences or prose
    counts as `json_valid=True` and `schema_valid=True`. Setting `allow_repair=True`
    attempts to strip markdown fences/prose for fallback diagnostics.
    """
    stripped = raw_text.strip()
    if not stripped:
        return ParseResult(
            raw_text=raw_text,
            json_valid=False,
            json_repaired=False,
            schema_valid=False,
            parsed_dict=None,
            validated=None,
            error="empty_output",
        )

    # 1. Strict JSON parse first
    try:
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            return ParseResult(
                raw_text=raw_text,
                json_valid=False,
                json_repaired=False,
                schema_valid=False,
                parsed_dict=None,
                validated=None,
                error="json_root_not_object",
            )
        repaired = False
    except json.JSONDecodeError as exc:
        if not allow_repair:
            return ParseResult(
                raw_text=raw_text,
                json_valid=False,
                json_repaired=False,
                schema_valid=False,
                parsed_dict=None,
                validated=None,
                error=f"json_decode_error: {exc.msg}",
            )
        candidate, repaired = extract_json_candidate(stripped)
        if candidate is None:
            return ParseResult(
                raw_text=raw_text,
                json_valid=False,
                json_repaired=False,
                schema_valid=False,
                parsed_dict=None,
                validated=None,
                error=f"json_decode_error: {exc.msg}",
            )
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                return ParseResult(
                    raw_text=raw_text,
                    json_valid=False,
                    json_repaired=repaired,
                    schema_valid=False,
                    parsed_dict=None,
                    validated=None,
                    error="json_root_not_object",
                )
        except json.JSONDecodeError as exc2:
            return ParseResult(
                raw_text=raw_text,
                json_valid=False,
                json_repaired=repaired,
                schema_valid=False,
                parsed_dict=None,
                validated=None,
                error=f"json_decode_error_after_repair: {exc2.msg}",
            )

    # 2. Strict Pydantic validation
    try:
        validated = TriageOutput.model_validate(obj)
        return ParseResult(
            raw_text=raw_text,
            json_valid=not repaired,
            json_repaired=repaired,
            schema_valid=True,
            parsed_dict=obj,
            validated=validated,
            error=None,
        )
    except ValidationError as exc:
        return ParseResult(
            raw_text=raw_text,
            json_valid=not repaired,
            json_repaired=repaired,
            schema_valid=False,
            parsed_dict=obj,
            validated=None,
            error=f"schema_validation_error: {exc.errors()[0]['loc']} - {exc.errors()[0]['msg']}",
        )
