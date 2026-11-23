# Benchmark Summary — `delivery-triage-llm-ptbr`

Avaliação determinística (`do_sample=False`, `seed=42`) no conjunto `test`.

| Modelo / Configuração | JSON Valid (%) | Schema Valid (%) | Macro-F1 (Categoria) | Macro-F1 (Sentimento) | Acc (Severidade) | Acc (Ação) | Latência p50 / p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-3B Base (Few-Shot, 3 ex) | 100.0% | 96.5% | 0.4460 | 0.5680 | 75.4% | 65.6% | 1657 / 1794 |
| Qwen2.5-3B + QLoRA v1 | 100.0% | 100.0% | 0.6908 | 0.6527 | 79.5% | 81.1% | 1671 / 1794 |
