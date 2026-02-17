"""Tests for enums and execution context."""

from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage
from cuvis_ai_schemas.execution import Context


def test_execution_stage_members():
    """ExecutionStage has exactly the expected members (no VALIDATE)."""
    expected = {"ALWAYS", "TRAIN", "VAL", "TEST", "INFERENCE"}
    actual = {m.name for m in ExecutionStage}
    assert actual == expected


def test_execution_stage_values():
    """ExecutionStage string values match expected."""
    assert ExecutionStage.ALWAYS == "always"
    assert ExecutionStage.TRAIN == "train"
    assert ExecutionStage.VAL == "val"
    assert ExecutionStage.TEST == "test"
    assert ExecutionStage.INFERENCE == "inference"


def test_artifact_type_members():
    """ArtifactType has the expected members."""
    assert ArtifactType.IMAGE == "image"


def test_context_defaults():
    """Context defaults to inference stage with zero counters."""
    ctx = Context()
    assert ctx.stage == ExecutionStage.INFERENCE
    assert ctx.epoch == 0
    assert ctx.batch_idx == 0
    assert ctx.global_step == 0


def test_context_to_dict():
    """Context.to_dict() produces expected dictionary."""
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=5, batch_idx=42, global_step=1337)
    d = ctx.to_dict()
    assert d == {
        "stage": "train",
        "epoch": 5,
        "batch_idx": 42,
        "global_step": 1337,
    }


def test_context_round_trip():
    """Context survives to_dict → from_dict round-trip."""
    original = Context(stage=ExecutionStage.VAL, epoch=10, batch_idx=3, global_step=500)
    restored = Context.from_dict(original.to_dict())
    assert restored.stage == original.stage
    assert restored.epoch == original.epoch
    assert restored.batch_idx == original.batch_idx
    assert restored.global_step == original.global_step
