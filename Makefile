.PHONY: lint lint-ci mypy
lint:
	uv run ruff format .
	uv run ruff check . --fix

lint-ci:
	uv run ruff check .
	uv run ruff format . --check

.PHONY: mypy
mypy:
	uv run mypy .

.PHONY: test
test:
	uv run pytest
