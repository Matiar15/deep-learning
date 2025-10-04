ruff:
	uv run ruff check --fix

mypy:
	uv run mypy .

lint: ruff mypy
