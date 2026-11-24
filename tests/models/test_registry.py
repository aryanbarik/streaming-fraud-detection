import pytest
from pydantic import ValidationError

from fraudstream.models.registry import load_model_config


@pytest.mark.parametrize("name", ["lgbm", "lr", "iforest"])
def test_packaged_configs_load_with_matching_kind(name):
    assert load_model_config(name).kind == name


def test_unknown_hyperparameter_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("kind: lgbm\nlearning_rat: 0.1\n")
    with pytest.raises(ValidationError):
        load_model_config("bad", tmp_path)


def test_missing_config_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        load_model_config("nope", tmp_path)
