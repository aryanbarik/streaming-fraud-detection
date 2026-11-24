"""Serving: one scoring interface over the trained model bundle and the rules baseline."""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from fraudstream.models.rules import RulesScorer as CountingRules
from fraudstream.models.train import ModelBundle
from fraudstream.stream.card_stage import FeatureRow
from fraudstream.stream.schemas import ScoredTx

# The rules baseline returns how many rules fired. Threshold 1.0 alerts when any rule fires.
RULES_THRESHOLD = 1.0


class Scorer(Protocol):
    def score(self, x: np.ndarray) -> float: ...


class LGBMScorer:
    """Scores one feature vector with a trained bundle. Feature order must match training."""

    def __init__(self, bundle: ModelBundle, feature_names: Sequence[str]) -> None:
        if bundle.kind != "lgbm":
            raise ValueError(f"LGBMScorer needs an lgbm bundle, got {bundle.kind!r}")
        if list(bundle.feature_list) != list(feature_names):
            raise ValueError("bundle feature_list does not match the serving feature order")
        self._bundle = bundle

    def score(self, x: np.ndarray) -> float:
        return float(self._bundle.score(x.reshape(1, -1))[0])


class RulesScorer:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self._rules = CountingRules(list(feature_names))

    def score(self, x: np.ndarray) -> float:
        return float(self._rules.score(x.reshape(1, -1))[0])


def build_scorer(
    kind: str, feature_names: Sequence[str], model_dir: Path | None = None
) -> tuple[Scorer, float]:
    """Return the scorer and its alert threshold. `kind` is `lgbm` or `rules`."""
    if kind == "lgbm":
        if model_dir is None:
            raise ValueError("the lgbm scorer needs model_dir")
        bundle = ModelBundle.load(model_dir)
        return LGBMScorer(bundle, feature_names), bundle.threshold
    if kind == "rules":
        return RulesScorer(feature_names), RULES_THRESHOLD
    raise ValueError(f"unknown scorer kind: {kind!r}")


def score_row(
    scorer: Scorer,
    row: FeatureRow,
    threshold: float,
    clock: Callable[[], int] = time.time_ns,
) -> ScoredTx:
    """Score one feature row. Alerts at `score >= threshold`, the convention used in evaluation."""
    score = scorer.score(row.features)
    t_score = clock()
    return ScoredTx(
        tx_id=row.tx_id,
        event_ts=row.event_ts,
        card_id=row.card_id,
        terminal_id=row.terminal_id,
        score=score,
        alert=score >= threshold,
        is_fraud=row.is_fraud,
        t_produce_ns=row.t_produce_ns,
        timings={**row.timings, "t_score": t_score, "t_emit": clock()},
    )
