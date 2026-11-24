"""Load named baseline configurations from configs/models/."""

from pathlib import Path

from fraudstream.config import ModelConfig, load_config

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"


def load_model_config(name: str, directory: Path = MODEL_CONFIG_DIR) -> ModelConfig:
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no model config {name!r} at {path}")
    return load_config(path, ModelConfig)
