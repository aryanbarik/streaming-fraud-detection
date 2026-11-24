"""Prometheus metrics for the streaming stages.

Names and labels are fixed: dashboards and reports read them by name. Each process builds its
metrics on the registry it serves, so nothing is global.
"""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Starting bucket edges; revisit after the first measured run.
STAGE_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)
E2E_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
SCORE_BUCKETS = tuple(round(0.05 * i, 2) for i in range(1, 21))


@dataclass(frozen=True)
class StreamMetrics:
    stage_latency: Histogram  # fs_stage_latency_seconds{stage}
    e2e_latency: Histogram  # fs_e2e_latency_seconds
    events: Counter  # fs_events_total{stage,status}
    consumer_lag: Gauge  # fs_consumer_lag_seconds{group}
    score_distribution: Histogram  # fs_score_distribution
    alert_rate: Gauge  # fs_alert_rate
    feature_psi: Gauge  # fs_feature_psi{feature}
    state_size: Gauge  # fs_state_size_bytes


def build_metrics(registry: CollectorRegistry) -> StreamMetrics:
    return StreamMetrics(
        stage_latency=Histogram(
            "fs_stage_latency_seconds",
            "Time inside one stage, per message.",
            ["stage"],
            registry=registry,
            buckets=STAGE_BUCKETS,
        ),
        e2e_latency=Histogram(
            "fs_e2e_latency_seconds",
            "Load generator send to scored output.",
            registry=registry,
            buckets=E2E_BUCKETS,
        ),
        events=Counter(
            "fs_events",
            "Messages handled, by stage and status.",
            ["stage", "status"],
            registry=registry,
        ),
        consumer_lag=Gauge(
            "fs_consumer_lag_seconds",
            "Consumer lag, by consumer group.",
            ["group"],
            registry=registry,
        ),
        score_distribution=Histogram(
            "fs_score_distribution",
            "Scores of scored transactions.",
            registry=registry,
            buckets=SCORE_BUCKETS,
        ),
        alert_rate=Gauge(
            "fs_alert_rate",
            "Share of recent scored transactions that alerted.",
            registry=registry,
        ),
        feature_psi=Gauge(
            "fs_feature_psi",
            "Population stability index, by feature.",
            ["feature"],
            registry=registry,
        ),
        state_size=Gauge(
            "fs_state_size_bytes",
            "Bytes held in stage state.",
            registry=registry,
        ),
    )
