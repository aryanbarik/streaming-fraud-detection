import numpy as np
import pytest

from fraudstream.config import DriftConfig, SimConfig
from fraudstream.sim.generator import generate
from fraudstream.sim.scenarios import Scenario


@pytest.mark.parametrize(
    ("name", "scenario"),
    [
        ("compromised_terminal", Scenario.COMPROMISED_TERMINAL),
        ("stolen_card", Scenario.STOLEN_CARD),
        ("card_testing", Scenario.CARD_TESTING),
        ("impossible_travel", Scenario.IMPOSSIBLE_TRAVEL),
    ],
)
def test_each_scenario_generates_fraud(name, scenario):
    names = ("compromised_terminal", "stolen_card", "card_testing", "impossible_travel")
    rates = {key: {"rate_per_day": float(key == name)} for key in names}
    tx = generate(SimConfig(n_customers=100, n_terminals=40, days=3, scenarios=rates), 9)
    scenarios = tx["scenario"].to_numpy()
    assert (scenarios == scenario).any()
    assert np.array_equal(tx["is_fraud"].to_numpy(), (scenarios != 0).astype(np.int8))
    if scenario == Scenario.CARD_TESTING:
        assert np.all(tx["amount"].to_numpy()[scenarios == scenario] <= 2)
    rows = tx.to_pylist()
    if scenario == Scenario.IMPOSSIBLE_TRAVEL:
        previous = {}
        for row in rows:
            if row["scenario"] == scenario:
                prior = previous[row["card_id"]]
                assert 0 <= (row["event_ts"] - prior["event_ts"]).total_seconds() <= 300
                assert np.hypot(row["x"] - prior["x"], row["y"] - prior["y"]) > 0.3
            previous[row["card_id"]] = row
    if scenario == Scenario.CARD_TESTING:
        bursts = {}
        for row in rows:
            if row["scenario"] == scenario:
                key = (row["card_id"], row["event_ts"].date())
                bursts.setdefault(key, []).append(row)
        assert any(len(group) >= 3 and
                   (group[2]["event_ts"] - group[0]["event_ts"]).total_seconds() <= 40
                   for group in bursts.values())


def test_benign_drift_raises_volume_only_in_window():
    base = SimConfig(n_customers=300, n_terminals=50, days=12)
    shifted = base.model_copy(update={"drift": DriftConfig(kind="benign_covariate", start_day=4, end_day=8)})
    tx = generate(base, 11)
    drift = generate(shifted, 11)
    def counts(table):
        dates = table["event_ts"].to_numpy().astype("datetime64[D]")
        return np.unique(dates, return_counts=True)[1]
    plain, altered = counts(tx), counts(drift)
    assert plain[0] == altered[0]
    assert altered[4:8].sum() > plain[4:8].sum() * 1.2
    for day in (*range(4), *range(8, 12)):
        date = np.datetime64("2025-01-01") + np.timedelta64(day, "D")
        a = tx.filter(tx["event_ts"].to_numpy().astype("datetime64[D]") == date)
        b = drift.filter(drift["event_ts"].to_numpy().astype("datetime64[D]") == date)
        assert a.drop(["tx_id"]).equals(b.drop(["tx_id"]))
