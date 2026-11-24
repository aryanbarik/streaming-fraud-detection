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
