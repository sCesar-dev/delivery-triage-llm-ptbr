.PHONY: install test lint format eval-report

install:
	uv sync --extra dev

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

eval-report:
	uv run python -m src.eval.evaluate --build-summary-only --results-dir results
