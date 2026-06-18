"""Data configuration schema: attributed sample universe + composable selectors."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel


class SampleRef(BaseSchemaModel):
    """One attributed sample in a DataModule's universe.

    A DataModule enumerates these; selectors pick subsets. ``uid`` is the stable,
    content-derived identity (``source#index``, plus ``#label_id`` when a row carries
    a distinct COCO id) and is the only key used for set algebra, leakage checks,
    dedup, and ordering. ``tags`` / ``category_ids`` are metadata and are never part
    of identity, so populating them lazily cannot change a sample's identity.
    """

    source: str = Field(description="Normalized asset path (cu3s / npz / tiff)")
    index: int | None = Field(
        default=None, description="Read position within source; None = whole-file sample"
    )
    label_id: int | None = Field(
        default=None, description="COCO image_id for annotation lookup; defaults to index"
    )
    stem: str = Field(default="", description="Filename stem (filled from source when empty)")
    annotation: str | None = Field(default=None, description="Paired json / label path")
    group: str | None = Field(default=None, description="Split-grouping key; defaults to source")
    tags: list[str] = Field(default_factory=list, description="Metadata tags (never identity)")
    category_ids: list[int] = Field(
        default_factory=list, description="Metadata category ids (never identity)"
    )
    uid: str = Field(default="", description="Stable content-derived identity (filled when empty)")

    @model_validator(mode="before")
    @classmethod
    def _fill_identity(cls, data: Any) -> Any:
        """Derive ``uid``/``group``/``stem`` from content when not supplied."""
        if isinstance(data, dict):
            source = data.get("source")
            if source is not None:
                index = data.get("index")
                label_id = data.get("label_id")
                if not data.get("stem"):
                    data["stem"] = Path(str(source)).stem
                if data.get("group") is None:
                    data["group"] = source
                if not data.get("uid"):
                    base = f"{source}#{index}" if index is not None else str(source)
                    if label_id is not None and label_id != index:
                        base = f"{base}#{label_id}"
                    data["uid"] = base
        return data


class SelectorKind(StrEnum):
    """The kinds of sample selector (a small, typed, composable query algebra)."""

    FILES = "files"
    FILE_INDICES = "file_indices"
    DIR_INDICES = "dir_indices"
    STEMS = "stems"
    GLOB = "glob"
    TAG = "tag"
    CATEGORIES = "categories"
    ALL = "all"
    UNION = "union"
    EXCEPT = "except"
    INTERSECT = "intersect"


#: Which ``Selector`` fields each kind may set; everything else must stay default.
_ALLOWED_FIELDS: dict[SelectorKind, set[str]] = {
    SelectorKind.FILES: {"paths"},
    SelectorKind.FILE_INDICES: {"source", "ids"},
    SelectorKind.DIR_INDICES: {"ids"},
    SelectorKind.STEMS: {"stems"},
    SelectorKind.GLOB: {"pattern"},
    SelectorKind.TAG: {"any_of"},
    SelectorKind.CATEGORIES: {"any_of"},
    SelectorKind.ALL: set(),
    SelectorKind.UNION: {"of"},
    SelectorKind.EXCEPT: {"of"},
    SelectorKind.INTERSECT: {"of"},
}

#: Fields each kind requires (ordered for stable error messages). For every
#: kind this equals its allowed set: a selector must set exactly the fields
#: valid for its kind, no more and no fewer.
_REQUIRED_FIELDS: dict[SelectorKind, tuple[str, ...]] = {
    SelectorKind.FILES: ("paths",),
    SelectorKind.FILE_INDICES: ("source", "ids"),
    SelectorKind.DIR_INDICES: ("ids",),
    SelectorKind.STEMS: ("stems",),
    SelectorKind.GLOB: ("pattern",),
    SelectorKind.TAG: ("any_of",),
    SelectorKind.CATEGORIES: ("any_of",),
    SelectorKind.ALL: (),
    SelectorKind.UNION: ("of",),
    SelectorKind.EXCEPT: ("of",),
    SelectorKind.INTERSECT: ("of",),
}

#: Scalar (single-value) fields; their absence reads "requires 'x'" rather than
#: "requires non-empty 'x'" (which suits the list-valued fields).
_SCALAR_FIELDS = {"source", "pattern"}

#: Set operations that need at least two operands.
_BINARY_SET_OPS = {SelectorKind.EXCEPT, SelectorKind.INTERSECT}


class Selector(BaseSchemaModel):
    """A composable sample selector over a ``SampleRef`` universe.

    Exactly the fields valid for ``kind`` may be set; everything else must stay at
    its default (structure-only validation). ``file_indices`` requires ``source``.
    Category / tag names in ``any_of`` are validated against the dataset at setup
    time, not here (the universe is not known at parse time).
    """

    kind: SelectorKind = Field(description="Selector kind")
    paths: list[str] = Field(default_factory=list, description="FILES: whole files by path")
    source: str | None = Field(
        default=None, description="FILE_INDICES: which file (required for file_indices)"
    )
    ids: list[int | str] = Field(
        default_factory=list,
        description="FILE_INDICES (within source) / DIR_INDICES (into the file list): int or 'a-b'",
    )
    stems: list[str] = Field(default_factory=list, description="STEMS: filename stems")
    pattern: str | None = Field(default=None, description="GLOB: a glob pattern")
    any_of: list[str] = Field(default_factory=list, description="TAG / CATEGORIES: names")
    of: list[Selector] = Field(
        default_factory=list, description="UNION / EXCEPT / INTERSECT operands"
    )

    @model_validator(mode="after")
    def _validate_structure(self) -> Selector:
        present = {
            "paths": bool(self.paths),
            "source": self.source is not None,
            "ids": bool(self.ids),
            "stems": bool(self.stems),
            "pattern": self.pattern is not None,
            "any_of": bool(self.any_of),
            "of": bool(self.of),
        }
        kind = self.kind.value

        # Only the fields valid for this kind may be set.
        for name, is_set in present.items():
            if is_set and name not in _ALLOWED_FIELDS[self.kind]:
                raise ValueError(f"selector kind '{kind}' must not set field '{name}'")

        # Every field required by this kind must be present.
        for field in _REQUIRED_FIELDS[self.kind]:
            if not present[field]:
                qualifier = "" if field in _SCALAR_FIELDS else "non-empty "
                raise ValueError(f"selector kind '{kind}' requires {qualifier}'{field}'")

        # except / intersect need at least two operands.
        if self.kind in _BINARY_SET_OPS and len(self.of) < 2:
            raise ValueError(f"selector kind '{kind}' requires at least 2 operands in 'of'")
        return self


class DataSplitConfig(BaseSchemaModel):
    """Split assignment as composable selectors over a ``SampleRef`` universe.

    Each stage is a list of selectors, unioned in order. A split assignment is
    committable to / evaluable from a ``splits.json`` (a serialized
    ``DataSplitConfig``). Empty ``predict`` means all samples.
    """

    splits_path: str | None = Field(
        default=None,
        description="Evaluate the assignment from this json (relative to the trainrun yaml dir)",
    )
    leakage_check: Literal["error", "warn", "off"] = Field(
        default="error",
        description="train/val/test disjointness: error raises, warn logs, off skips",
    )
    universe_hash: str | None = Field(
        default=None,
        description="Ordered file-list fingerprint; verified on load when dir_indices remain",
    )
    train: list[Selector] = Field(default_factory=list, description="Training selectors (unioned)")
    val: list[Selector] = Field(default_factory=list, description="Validation selectors (unioned)")
    test: list[Selector] = Field(default_factory=list, description="Test selectors (unioned)")
    predict: list[Selector] = Field(
        default_factory=list, description="Predict selectors; empty -> all samples"
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


__all__ = ["DataConfig", "DataSplitConfig", "SampleRef", "SelectorKind", "Selector"]
