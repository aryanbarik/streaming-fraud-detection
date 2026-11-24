"""Restrict training labels to reports available at a given cutoff."""

from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class LabelView:
    """A transaction-aligned table containing nullable report availability timestamps.

    A null availability timestamp represents an unreported fraud or a legitimate
    transaction.  Ground-truth columns are deliberately not inspected.
    """

    def __init__(self, table: pa.Table, as_of: datetime):
        self.table = table
        self.as_of = as_of

    def labels(self) -> np.ndarray:
        visible = pc.fill_null(pc.less_equal(self.table["available_ts"], self.as_of), False)
        return visible.to_numpy().astype(np.int8)

    def mature(self, days: int = 14) -> np.ndarray:
        visible = pc.fill_null(pc.less_equal(self.table["event_ts"], self.as_of - timedelta(days=days)), False)
        return visible.to_numpy().astype(bool)
