# 0005: Sparkov adapter

Status: accepted (P3). Dataset facts (`kartik2112/fraud-detection`, file names, column meanings,
license) are unverified until the first real download; the adapter is tested on a fixture only.

## Context

Sparkov is the secondary dataset (ADR 0001). It has no label delays and no channel, scenario, or
category-level MCC fields that the feature backfill reads. This environment has no Kaggle token,
so the Sparkov column of E1 is omitted until one exists.

## Decision

- `fraudTrain.csv` and `fraudTest.csv` are concatenated and sorted by `unix_time` (stable sort).
- `event_ts = 2025-01-01 + (unix_time - min unix_time)`, so day 0 matches the simulator origin
  (`fraudstream.timeline.ORIGIN`).
- `card_id` = factorized `cc_num`; `terminal_id` = factorized `merchant`; `amount` = `amt`;
  `is_fraud` = `is_fraud`.
- `x`, `y` = merchant longitude and latitude min-max scaled over the whole file. Cardholder
  `lat`/`long` are fixed per card, so they cannot drive travel features. The scaling uses no
  labels.
- Labels are simulated with the simulator's delay model (`LabelConfig()` defaults, seed + 1)
  because Sparkov reports no delays. Sparkov results therefore measure models under assumed
  delays, not under Sparkov's true delays.
- `mcc`, `channel`, and `scenario` are not written: the feature backfill does not read them, and
  filling them with constants would invent signal.

## Consequences

- `make data-sparkov` runs only when `~/.kaggle/kaggle.json` exists; otherwise it prints
  `skipped: no Kaggle token` and exits 0.
- Sparkov E1 and E4 results are produced only after the download and backfill succeed.
