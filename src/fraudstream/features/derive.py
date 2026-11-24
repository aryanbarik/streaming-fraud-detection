"""Compute numerical features from pre-update state."""

import math

from fraudstream.features.state import Tx


def derive(tx: Tx, card_feats: dict, term_feats: dict) -> dict[str, float]:
    mean = card_feats["card_amount_mean_7d"]
    std = card_feats["card_amount_std_7d"]
    previous = card_feats["card_last_ts"]
    last_xy = card_feats["card_last_xy"]
    speed = 0.0
    if previous is not None and last_xy is not None:
        hours = (tx.event_ts - previous).total_seconds() / 3600
        if hours > 0:
            speed = math.hypot(tx.x - last_xy[0], tx.y - last_xy[1]) / hours
    result = {key: value for key, value in card_feats.items() if key not in {"card_last_ts", "card_last_xy"}}
    result.update(term_feats)
    result.update({
        "amount_mean_ratio": tx.amount / mean if mean else 0.0,
        "amount_zscore": (tx.amount - mean) / std if std else 0.0,
        "hour": tx.event_ts.hour,
        "weekday": tx.event_ts.weekday(),
        "travel_units_per_hour": speed,
    })
    return result
