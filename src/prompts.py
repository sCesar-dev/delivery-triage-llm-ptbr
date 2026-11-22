"""Versioned system prompts and ChatML message builders (Zero-Shot & Few-Shot)."""

from __future__ import annotations

import json

PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT = """Você é um sistema especializado de triagem operacional de reviews e incidentes de pedidos de delivery em português brasileiro.
Analise a mensagem do cliente e retorne EXCLUSIVAMENTE um objeto JSON válido, sem markdown, sem crases e sem explicações extras, seguindo exatamente este contrato:
{
  "sentimento": "positivo" | "neutro" | "negativo",
  "categoria_incidente": "entrega_atrasada" | "item_errado_ou_faltando" | "qualidade_produto" | "atendimento" | "pagamento_cobranca" | "elogio_geral" | "outro",
  "severidade": "baixa" | "media" | "alta" | "critica",
  "requer_acao_humana": true | false,
  "acao_sugerida": "nenhuma" | "acompanhar" | "estorno_parcial" | "estorno_total" | "escalar_suporte",
  "resumo_curto": "Frase objetiva em português de até 15 palavras descrevendo o ponto central."
}"""

FEW_SHOT_EXAMPLES: list[tuple[str, dict]] = [
    (
        "Pedido atrasou mais de 1 hora e chegou completamente frio. Ninguém responde no chat!",
        {
            "sentimento": "negativo",
            "categoria_incidente": "entrega_atrasada",
            "severidade": "alta",
            "requer_acao_humana": True,
            "acao_sugerida": "escalar_suporte",
            "resumo_curto": "Entrega atrasou mais de uma hora e comida chegou fria sem suporte.",
        },
    ),
    (
        "Pedi dois lanches e uma coca, mas o refrigerante não veio na sacola.",
        {
            "sentimento": "negativo",
            "categoria_incidente": "item_errado_ou_faltando",
            "severidade": "media",
            "requer_acao_humana": True,
            "acao_sugerida": "estorno_parcial",
            "resumo_curto": "Refrigerante faltando no pedido entregue ao cliente.",
        },
    ),
    (
        "Chegou antes do prazo, super bem embalado e quentinho. Recomendo demais!",
        {
            "sentimento": "positivo",
            "categoria_incidente": "elogio_geral",
            "severidade": "baixa",
            "requer_acao_humana": False,
            "acao_sugerida": "nenhuma",
            "resumo_curto": "Cliente elogia rapidez na entrega, embalagem e temperatura do pedido.",
        },
    ),
]


def build_user_prompt(review_text: str) -> str:
    """Format a customer review/complaint as the user message."""
    return f"Review do pedido:\n{review_text.strip()}"


def build_messages(
    review_text: str,
    mode: str = "zero_shot",
    system_prompt: str = SYSTEM_PROMPT,
    target_json: str | dict | None = None,
) -> list[dict[str, str]]:
    """Build ChatML-compatible messages list for inference or SFT training.

    Args:
        review_text: Raw PT-BR review or complaint text.
        mode: Either "zero_shot" or "few_shot".
        system_prompt: System prompt defining the strict JSON schema.
        target_json: Optional ground-truth assistant response (for SFT training).
    """
    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError(f"Unsupported prompt mode: {mode!r}. Expected 'zero_shot' or 'few_shot'.")

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if mode == "few_shot":
        for example_text, example_output in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": build_user_prompt(example_text)})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example_output, ensure_ascii=False),
                }
            )

    messages.append({"role": "user", "content": build_user_prompt(review_text)})

    if target_json is not None:
        assistant_content = (
            target_json
            if isinstance(target_json, str)
            else json.dumps(target_json, ensure_ascii=False)
        )
        messages.append({"role": "assistant", "content": assistant_content})

    return messages
