import numpy as np

from fraudstream.config import LabelConfig, SimConfig
from fraudstream.sim.generator import generate
from fraudstream.sim.labels import make_label_events


def test_determinism_and_event_order():
    cfg = SimConfig(n_customers=40, n_terminals=20, days=3)
    a = generate(cfg, 7)
    b = generate(cfg, 7)
    assert a.equals(b)
    times = a["event_ts"].to_numpy()
    assert np.all(times[1:] >= times[:-1])
    assert len(np.unique(a["tx_id"].to_numpy())) == a.num_rows


def test_reported_labels_follow_transactions():
    cfg = SimConfig(n_customers=80, n_terminals=30, days=10,
                    scenarios={"card_testing": {"rate_per_day": 0.2}})
    tx = generate(cfg, 2)
    labels = make_label_events(tx, LabelConfig(p_report=1), 3)
    assert labels.num_rows == int(tx["is_fraud"].to_numpy().sum())
    ids = labels["tx_id"].to_numpy()
    assert np.all(labels["available_ts"].to_numpy() > tx["event_ts"].to_numpy()[ids])
    assert np.array_equal(labels["terminal_id"].to_numpy(), tx["terminal_id"].to_numpy()[ids])
