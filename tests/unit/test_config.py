"""Unit tests for configuration parsing, YAML loading, and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.utils.config import (
    AppConfig,
    get_app_config,
    load_app_config,
)


def test_load_app_config_from_yaml_dir() -> None:
    """Test loading configs from repository ./configs directory."""
    config_dir = Path("./configs").resolve()
    assert config_dir.is_dir()

    app_cfg = load_app_config(config_dir=config_dir)
    assert isinstance(app_cfg, AppConfig)
    assert app_cfg.models.generator.model_id is not None
    assert app_cfg.models.verifier.model_id is not None
    assert app_cfg.retrieval.bm25.k1 > 0.0
    assert len(app_cfg.experiments) > 0


def test_get_experiment_by_id() -> None:
    """Test fetching defined experiment specifications."""
    app_cfg = get_app_config(reload=True)
    exp_e5 = app_cfg.get_experiment("E5_raft")

    assert exp_e5.experiment_id == "E5_raft"
    assert exp_e5.generator == "raft"
    assert exp_e5.retrieval == "hybrid"
    assert exp_e5.verification is False

    exp_e7 = app_cfg.get_experiment("E7_with_verification")
    assert exp_e7.verification is True

    # Unknown experiment should raise KeyError
    with pytest.raises(KeyError):
        app_cfg.get_experiment("non_existent_experiment_xyz")


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables override default configuration paths and parameters."""
    monkeypatch.setenv("MEDFUSION_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("MEDFUSION_DATA_DIR", "/custom/data/path")

    app_cfg = load_app_config()
    assert app_cfg.env.log_level == "DEBUG"
    assert app_cfg.env.cuda_visible_devices == "2,3"
    assert app_cfg.env.data_dir == "/custom/data/path"


def test_singleton_config_caching() -> None:
    """Test get_app_config returns cached singleton unless reload=True."""
    cfg1 = get_app_config()
    cfg2 = get_app_config()
    assert cfg1 is cfg2

    cfg3 = get_app_config(reload=True)
    assert cfg3 is not None
