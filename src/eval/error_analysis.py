"""Summarize error patterns from `results/error_cases_*.jsonl` for qualitative review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize_errors(error_file: Path, top_n: int = 10) -> str:
    """Produce a concise markdown summary of failure modes and top confusion pairs."""
    if not error_file.exists():
        raise FileNotFoundError(f"Error file not found: {error_file}")

    rows = [
        json.loads(line)
        for line in error_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return "No error cases found."

    failure_types = Counter()
    confusion_pairs = Counter()

    for row in rows:
        if not row.get("json_valid"):
            failure_types["invalid_json"] += 1
        elif not row.get("schema_valid"):
            failure_types["invalid_schema"] += 1
        else:
            gt = row["ground_truth"]
            pred = row["predicted"] or {}
            if pred.get("categoria_incidente") != gt.get("categoria_incidente"):
                failure_types["categoria_mismatch"] += 1
                confusion_pairs[
                    (gt.get("categoria_incidente"), pred.get("categoria_incidente"))
                ] += 1
            if pred.get("sentimento") != gt.get("sentimento"):
                failure_types["sentimento_mismatch"] += 1
            if pred.get("severidade") != gt.get("severidade"):
                failure_types["severidade_mismatch"] += 1
            if pred.get("acao_sugerida") != gt.get("acao_sugerida"):
                failure_types["acao_mismatch"] += 1

    lines = [
        f"## Error Analysis Summary (`{error_file.name}` — {len(rows)} cases)",
        "",
        "### Failure Mode Counts",
    ]
    for mode, count in failure_types.most_common():
        lines.append(f"- `{mode}`: {count}")

    if confusion_pairs:
        lines.extend(["", "### Top Category Confusions (Ground Truth → Predicted)"])
        for (gt_cat, pred_cat), count in confusion_pairs.most_common(top_n):
            lines.append(f"- `{gt_cat}` → `{pred_cat}`: {count}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation error cases")
    parser.add_argument("--error-file", type=str, default="results/error_cases.jsonl")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    print(summarize_errors(Path(args.error_file), top_n=args.top_n))


if __name__ == "__main__":
    main()
