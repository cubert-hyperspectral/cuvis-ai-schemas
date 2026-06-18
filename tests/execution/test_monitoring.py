"""Tests for the Artifact and Metric monitoring dataclasses."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage
from cuvis_ai_schemas.execution import Artifact, Metric


def test_metric_defaults():
    """Metric defaults to the inference stage at epoch/batch 0."""
    m = Metric(name="loss/train", value=0.5)
    assert m.value == pytest.approx(0.5)
    assert m.stage == ExecutionStage.INFERENCE
    assert m.epoch == 0
    assert m.batch_idx == 0


def test_metric_explicit_fields():
    """Metric stores explicit stage/epoch/batch values."""
    m = Metric(name="loss", value=0.1, stage=ExecutionStage.TRAIN, epoch=2, batch_idx=3)
    assert (m.stage, m.epoch, m.batch_idx) == (ExecutionStage.TRAIN, 2, 3)


def test_artifact_construction():
    """Artifact carries its array value, type, and default stage."""
    np = pytest.importorskip("numpy")
    arr = np.zeros((4, 4, 3))
    artifact = Artifact(name="heatmap", value=arr, el_id=0, desc="d", type=ArtifactType.IMAGE)
    assert artifact.type == ArtifactType.IMAGE
    assert artifact.stage == ExecutionStage.INFERENCE
    assert artifact.value.shape == (4, 4, 3)
