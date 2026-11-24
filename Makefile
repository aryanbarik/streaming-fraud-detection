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

backfill-tiny:
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.features.backfill --transactions data/sim/seed0/transactions.parquet --labels data/sim/seed0/labels.parquet --out data/sim/seed0/features.parquet

e4-tiny:
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.experiments.e4_imbalance --data data/sim/seed0 --out results/e4_tiny/seed0

e1-tiny: e4-tiny
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.experiments.e1_models --data data/sim/seed0 --out results/e1_tiny/seed0 --imbalance-from results/e4_tiny/seed0/metrics.json

e3-tiny:
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.experiments.e3_leakage --data data/sim/seed0 --out results/e3_tiny/seed0

e4b-tiny: e4-tiny
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run python -m fraudstream.experiments.e4b_threshold --data data/sim/seed0 --out results/e4b_tiny/seed0 --imbalance-from results/e4_tiny/seed0/metrics.json

data-sparkov:
	@if [ ! -f "$$HOME/.kaggle/kaggle.json" ]; then echo 'skipped: no Kaggle token'; exit 0; fi
	@free=$$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); \
	  test "$$free" -ge 16 || { echo 'disk guard: need at least 16 GB free'; exit 1; }
	uv run --with kaggle==2.2.5 kaggle datasets download -d kartik2112/fraud-detection -p data/sparkov/raw --unzip
	uv run python -m fraudstream.datasets.sparkov --raw data/sparkov/raw --out data/sparkov/seed0 --seed 0
	uv run python -m fraudstream.features.backfill --transactions data/sparkov/seed0/transactions.parquet --labels data/sparkov/seed0/labels.parquet --out data/sparkov/seed0/features.parquet

e4-sparkov:
	@if [ ! -f data/sparkov/seed0/features.parquet ]; then echo 'skipped: no Sparkov features (run make data-sparkov)'; exit 0; fi
	uv run python -m fraudstream.experiments.e4_imbalance --data data/sparkov/seed0 --out results/e4_sparkov/seed0

e1-sparkov: e4-sparkov
	@if [ ! -f data/sparkov/seed0/features.parquet ]; then echo 'skipped: no Sparkov features (run make data-sparkov)'; exit 0; fi
	uv run python -m fraudstream.experiments.e1_models --data data/sparkov/seed0 --out results/e1_sparkov/seed0 --imbalance-from results/e4_sparkov/seed0/metrics.json
