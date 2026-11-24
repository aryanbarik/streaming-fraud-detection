"""Seeded synthetic transaction histories, with persistent fraud campaigns."""

from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa

from fraudstream.config import SimConfig
from fraudstream.sim.scenarios import Scenario

_SCHEMA = pa.schema([
    ("tx_id", pa.int64()), ("event_ts", pa.timestamp("us")),
    ("card_id", pa.int32()), ("terminal_id", pa.int32()),
    ("amount", pa.float32()), ("mcc", pa.int8()), ("channel", pa.int8()),
    ("x", pa.float32()), ("y", pa.float32()), ("is_fraud", pa.int8()),
    ("scenario", pa.int8()),
])


def _rate(cfg: SimConfig, name: str, default: float) -> float:
    value = float(cfg.scenarios.get(name, {}).get("rate_per_day", default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} rate_per_day must be in [0, 1]")
    return value


def generate(cfg: SimConfig, seed: int) -> pa.Table:
    """Generate chronological transactions; campaigns persist across days."""
    rng = np.random.default_rng(seed)
    card_xy = rng.random((cfg.n_customers, 2))
    terminal_xy = rng.random((cfg.n_terminals, 2))
    terminal_mcc = rng.integers(0, 64, cfg.n_terminals, dtype=np.int8)
    card_amount = rng.lognormal(3.5, 0.5, cfg.n_customers)
    card_rate = rng.uniform(0, 4, cfg.n_customers)
    # Sample uniformly from nearby terminals, or use the nearest if none exist.
    eligible = []
    for position in card_xy:
        dist2 = ((terminal_xy - position) ** 2).sum(axis=1)
        nearby = np.flatnonzero(dist2 <= cfg.radius ** 2)
        eligible.append(nearby if len(nearby) else np.array([dist2.argmin()]))
    compromised_until = np.zeros(cfg.n_terminals, dtype=np.int32)
    stolen_until = np.zeros(cfg.n_customers, dtype=np.int32)
    rates = {
        "compromised_terminal": _rate(cfg, "compromised_terminal", 0.0015),
        "stolen_card": _rate(cfg, "stolen_card", 0.0001),
        "card_testing": _rate(cfg, "card_testing", 0.001),
        "impossible_travel": _rate(cfg, "impossible_travel", 0.001),
    }
    chunks = []
    next_id = 0
    start = datetime(2025, 1, 1)
    for day in range(cfg.days):
        drift = cfg.drift
        active = drift.start_day <= day < drift.end_day
        rng = np.random.default_rng(np.random.SeedSequence([seed, day, 1]))
        campaign_rng = np.random.default_rng(np.random.SeedSequence([seed, day, 2]))
        volume = 1.4 if active and drift.kind == "benign_covariate" else 1.0
        n_by_card = rng.poisson(card_rate * volume)
        cards = np.repeat(np.arange(cfg.n_customers, dtype=np.int32), n_by_card)
        n = len(cards)
        compromise_rate = rates["compromised_terminal"]
        travel_rate = rates["impossible_travel"]
        if drift.kind == "abrupt_concept" and day >= drift.start_day:
            compromise_rate *= 0.5
            travel_rate = min(1.0, travel_rate * 3)
        if drift.kind == "gradual_prior" and day >= drift.start_day:
            span = max(1, drift.end_day - drift.start_day)
            factor = 1 + 2 * min(1, (day - drift.start_day) / span)
            compromise_rate = min(1.0, compromise_rate * factor)
            travel_rate = min(1.0, travel_rate * factor)
        newly_compromised = campaign_rng.random(cfg.n_terminals) < compromise_rate
        newly_stolen = campaign_rng.random(cfg.n_customers) < rates["stolen_card"]
        compromised_until[newly_compromised] = day + 28
        stolen_until[newly_stolen] = day + 14
        terminals = np.fromiter(
            (rng.choice(eligible[card]) for card in cards), dtype=np.int32, count=n
        )
        micros = rng.integers(0, 86_400_000_000, n, dtype=np.int64)
        scenario = np.zeros(n, dtype=np.int8)
        scenario[(compromised_until[terminals] > day) & (rng.random(n) < 0.05)] = Scenario.COMPROMISED_TERMINAL
        scenario[(stolen_until[cards] > day) & (rng.random(n) < 1 / 3)] = Scenario.STOLEN_CARD
        offsets = np.cumsum(n_by_card) - n_by_card
        eligible_travel = np.flatnonzero(n_by_card >= 2)
        travel_cards = eligible_travel[rng.random(len(eligible_travel)) < travel_rate]
        for card in travel_cards:
            first = offsets[card]
            second = first + 1
            micros[first] = rng.integers(0, 86_100_000_000)
            micros[second] = micros[first] + 300_000_000
            distance = ((terminal_xy - terminal_xy[terminals[first]]) ** 2).sum(axis=1)
            terminals[second] = int(distance.argmax())
            scenario[second] = Scenario.IMPOSSIBLE_TRAVEL
        testing_candidates = np.flatnonzero(n_by_card >= 3)
        testing_candidates = testing_candidates[~np.isin(testing_candidates, travel_cards)]
        testing_cards = testing_candidates[
            rng.random(len(testing_candidates)) < rates["card_testing"]
        ]
        for card in testing_cards:
            first = offsets[card]
            micros[first] = rng.integers(0, 86_340_000_000)
            micros[first + 1] = micros[first] + 20_000_000
            micros[first + 2] = micros[first] + 40_000_000
            scenario[first:first + 3] = Scenario.CARD_TESTING
            terminals[first:first + 3] = terminals[first]
        amount = card_amount[cards] * rng.lognormal(0, 0.5, n)
        amount[scenario == Scenario.STOLEN_CARD] *= 5
        amount[scenario == Scenario.CARD_TESTING] = rng.uniform(0.5, 2.0, (scenario == Scenario.CARD_TESTING).sum())
        if active and drift.kind == "benign_covariate":
            amount *= 1.5
        order = np.argsort(micros, kind="stable")
        event_ts = np.datetime64(start + timedelta(days=day), "us") + micros[order].astype("timedelta64[us]")
        cards, terminals, scenario, amount = cards[order], terminals[order], scenario[order], amount[order]
        data = [np.arange(next_id, next_id + n), event_ts, cards, terminals, amount.astype(np.float32),
                terminal_mcc[terminals], rng.integers(0, 2, n, dtype=np.int8),
                terminal_xy[terminals, 0].astype(np.float32), terminal_xy[terminals, 1].astype(np.float32),
                (scenario != 0).astype(np.int8), scenario]
        chunks.append(pa.Table.from_arrays(data, schema=_SCHEMA))
        next_id += n
    return pa.concat_tables(chunks)
