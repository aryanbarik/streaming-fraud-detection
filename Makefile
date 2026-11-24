.PHONY: setup test lint
setup:
	uv sync --all-groups
test:
	uv run pytest -m 'not network and not slow and not gpu and not docker'
lint:
	uv run ruff check

data-tiny:
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.sim.cli --config configs/sim_tiny.yaml --seed 0 --out data/sim/seed0
