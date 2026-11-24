"""Names and types of the numeric features used for scoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    entity: str
    window: str


FEATURES = [
    FeatureSpec("card_count_1h", "int64", "card", "1h"),
    FeatureSpec("card_count_24h", "int64", "card", "24h"),
    FeatureSpec("card_count_7d", "int64", "card", "7d"),
    FeatureSpec("card_amount_mean_7d", "float64", "card", "7d"),
    FeatureSpec("card_amount_std_7d", "float64", "card", "7d"),
    FeatureSpec("card_distinct_terminals_7d", "int64", "card", "7d"),
    FeatureSpec("terminal_count_1h", "int64", "terminal", "1h"),
    FeatureSpec("terminal_count_24h", "int64", "terminal", "24h"),
    FeatureSpec("terminal_count_7d", "int64", "terminal", "7d"),
    FeatureSpec("terminal_fraud_reports_7d", "int64", "terminal", "7d"),
    FeatureSpec("terminal_report_rate_7d", "float64", "terminal", "7d"),
    FeatureSpec("amount_mean_ratio", "float64", "derived", "7d"),
    FeatureSpec("amount_zscore", "float64", "derived", "7d"),
    FeatureSpec("hour", "int64", "derived", "instant"),
    FeatureSpec("weekday", "int64", "derived", "instant"),
    FeatureSpec("travel_units_per_hour", "float64", "derived", "previous"),
]
