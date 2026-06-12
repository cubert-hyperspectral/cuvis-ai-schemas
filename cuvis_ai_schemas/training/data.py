"""Data configuration schema."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel


class DataSplitConfig(BaseSchemaModel):
    """Universal split assignment, validated for every DataModule.

    Each entry is a sample *selector*: an ``int`` (positional / measurement
    index, e.g. cu3s), a ``str`` key (e.g. a TIFF filename stem), or an inclusive
    range string ``"start-stop[:step]"`` (e.g. ``"0-100"`` or ``"0-10:2"``) that
    the base expands to integer selectors before ``build_dataset(ids)``. The
    DataModule resolves selectors to samples in ``build_dataset(ids)``. A module
    whose split assignment cannot be expressed as a flat selector list (e.g. a
    CSV-encoded multi-file split) leaves ``DataConfig.splits`` as ``None`` and
    owns its split logic in ``build_stage_dataset(stage)``.
    """

    train_ids: list[int | str] = Field(
        default_factory=list, description="Training sample selectors"
    )
    val_ids: list[int | str] = Field(
        default_factory=list, description="Validation sample selectors"
    )
    test_ids: list[int | str] = Field(default_factory=list, description="Test sample selectors")
    predict_ids: list[int | str] = Field(
        default_factory=list, description="Predict sample selectors; empty -> all samples"
    )


class DataConfig(BaseSchemaModel):
    """Polymorphic data-loading configuration.

    ``data_module`` selects a registered DataModule by its ``DATA_MODULE_NAME``;
    ``params`` carries the module-specific arguments. The shape is module-agnostic
    so any plugin DataModule (cu3s, tiff_paired, ...) can be expressed.
    """

    __proto_message__: ClassVar[str] = "DataConfig"

    data_module: str = Field(default="cu3s", description="DataModule name (its DATA_MODULE_NAME)")
    splits: DataSplitConfig | None = Field(
        default=None,
        description="Split selectors; None for predict/inference or module-owned splits",
    )
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    num_workers: int = Field(default=0, ge=0, description="DataLoader worker processes")
    params: dict[str, Any] = Field(default_factory=dict, description="Module-specific arguments")


class SplitsResolveConfig(BaseSchemaModel):
    """Request payload for the ``ResolveSplits`` RPC (workspace split resolution).

    Carried as JSON in ``ResolveSplitsRequest.config_bytes``. ``data_module``
    names the registered DataModule whose ``resolve_splits`` classmethod owns the
    strategy semantics (e.g. ``cu3s_workspace``: anomaly-aware random/stratified
    with train = normal frames only), so future task types plug in without a
    proto change. ``seed=None`` defers to the workspace default seed.
    """

    workspace_path: str = Field(description="Workspace folder (holds workspace.json)")
    data_module: str = Field(
        default="cu3s_workspace", description="DataModule that resolves the splits"
    )
    strategy: str = Field(
        default="random", pattern="^(random|stratified)$", description="Split strategy"
    )
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0, description="Train share")
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Val share")
    seed: int | None = Field(default=None, description="RNG seed; None -> workspace default")
    selected_files: list[str] | None = Field(
        default=None, description="Member subset to resolve; None -> all members"
    )
    write: bool = Field(default=True, description="Write splits.json into the workspace")


__all__ = ["DataConfig", "DataSplitConfig", "SplitsResolveConfig"]
