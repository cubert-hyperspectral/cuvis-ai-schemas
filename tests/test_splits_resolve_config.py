"""Tests for SplitsResolveConfig (the ResolveSplits RPC payload)."""

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training.data import SplitsResolveConfig


def test_defaults_and_roundtrip():
    cfg = SplitsResolveConfig(workspace_path="/ws")
    assert cfg.data_module == "cu3s_workspace"
    assert cfg.strategy == "random"
    assert cfg.train_ratio == 0.70 and cfg.val_ratio == 0.15
    assert cfg.seed is None and cfg.selected_files is None and cfg.write is True
    again = SplitsResolveConfig.from_json(cfg.to_json())
    assert again == cfg


def test_strategy_is_constrained():
    SplitsResolveConfig(workspace_path="/ws", strategy="stratified")
    with pytest.raises(ValidationError):
        SplitsResolveConfig(workspace_path="/ws", strategy="bogus")


def test_ratio_bounds():
    with pytest.raises(ValidationError):
        SplitsResolveConfig(workspace_path="/ws", train_ratio=0.0)
    with pytest.raises(ValidationError):
        SplitsResolveConfig(workspace_path="/ws", val_ratio=1.0)


def test_proto_messages_exist():
    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

    req = cuvis_ai_pb2.ResolveSplitsRequest(session_id="s", config_bytes=b"{}")
    resp = cuvis_ai_pb2.ResolveSplitsResponse(splits_bytes=b"{}", splits_path="/ws/splits.json")
    assert req.session_id == "s" and resp.splits_path.endswith("splits.json")
