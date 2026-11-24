# 0004: Model bundle format

Status: accepted (P3)

## Context

Appendix A asks for `model.txt` serving artifacts. The P3 code saved every kind with joblib and
recorded calibration as the literal string `"none"`. Calibration must be stored with its
parameters so a serving process can reproduce scores without refitting.

## Decision

- LightGBM bundles save the native booster as `model.txt` and load it with
  `lgb.Booster(model_file=...)`. The serving path (E7 arms, all LightGBM) never unpickles.
- lr and iforest keep `model.joblib`; sklearn estimators have no text format in the pinned
  versions. This is the only deviation from the single `model.txt` artifact in Appendix A.
- `manifest.json` records calibration as an object: `{"method": "none"}` or
  `{"method": "platt", "coef": ..., "intercept": ...}`. Platt parameters are fit only on
  validation rows through `ModelBundle.calibrate`.
- `train` takes a `ModelConfig` whose `kind` names the model, plus `feature_list`, instead of
  the `train(model: str, X, y, cfg)` interface in `docs/project-plan.md`. `load_model_config(name)`
  reads `configs/models/<name>.yaml`; the model name is never passed twice.

## Consequences

- Manifests written before this change (string calibration field) do not load. None were
  committed as artifacts; results directories hold metrics only.
