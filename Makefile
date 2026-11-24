.PHONY: setup test lint
setup:
	uv sync --all-groups
test:
	uv run pytest -m 'not network and not slow and not gpu and not docker'
lint:
	uv run ruff check
