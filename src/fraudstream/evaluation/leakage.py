"""Intentionally leaky feature variants for leakage experiments only."""

import pandas as pd
import pyarrow as pa


def immediate_fraud_labels(transactions: pa.Table) -> pa.Table:
    """Expose ground truth at transaction time, deliberately violating label delay."""
    fraud = transactions.filter(transactions["is_fraud"].to_numpy() == 1)
    return pa.table({
        "tx_id": fraud["tx_id"],
        "terminal_id": fraud["terminal_id"],
        "available_ts": fraud["event_ts"],
        "kind": pa.array(["fraud"] * fraud.num_rows, type=pa.string()),
    })


def future_terminal_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace terminal features with the last observed state, leaking future data."""
    ordered = frame.sort_values(["event_ts", "tx_id"], kind="stable")
    latest = ordered.groupby("terminal_id", sort=False).tail(1).set_index("terminal_id")
    result = frame.copy()
    result[columns] = latest.loc[result["terminal_id"], columns].to_numpy()
    return result
