"""Configuration models for simulation, costs and streaming."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LabelConfig(BaseModel):
    p_report: float = Field(default=0.85, ge=0, le=1)
    median_delay_days: float = Field(default=7, gt=0)
    sigma: float = Field(default=0.8, ge=0)


class DriftConfig(BaseModel):
    kind: str = "none"
    start_day: int = 100
    end_day: int = 130
    magnitude: float = 1.0


class SimConfig(BaseModel):
    n_customers: int = Field(default=10_000, gt=0)
    n_terminals: int = Field(default=3_000, gt=0)
    days: int = Field(default=180, gt=0)
    radius: float = Field(default=0.1, gt=0)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    scenarios: dict = Field(default_factory=dict)


class CostConfig(BaseModel):
    review_usd: float = Field(default=5, ge=0)
    fp_friction_usd: float = Field(default=3, ge=0)
    fraud_fixed_fee_usd: float = Field(default=0, ge=0)


class StreamConfig(BaseModel):
    broker: str = "localhost:9092"
    deadline_ms: float = Field(default=100, gt=0)


def load_config(path: Path, model: type[BaseModel]) -> BaseModel:
    with path.open() as stream:
        return model.model_validate(yaml.safe_load(stream))
