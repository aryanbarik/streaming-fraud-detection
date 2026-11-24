# Point-in-time feature consistency

Share the feature-state implementation between streaming and backfill. Evaluate Feast's point-in-time join and online push before adopting them; if the spike fails, use a DuckDB ASOF join and retain the skew test.
