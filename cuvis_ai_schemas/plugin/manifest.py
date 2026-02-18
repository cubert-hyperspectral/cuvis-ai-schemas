"""Plugin manifest schema."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field, field_validator

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.plugin.config import GitPluginConfig, LocalPluginConfig


class PluginManifest(BaseSchemaModel):
    """Complete plugin manifest containing all plugin configurations.

    This is the root configuration object validated when loading
    a plugins.yaml file or dictionary.
    """

    plugins: dict[
        str,
        Annotated[
            GitPluginConfig | LocalPluginConfig,
            Field(discriminator=None),  # Pydantic will auto-detect based on fields
        ],
    ] = Field(
        description="Map of plugin names to their configurations",
        default_factory=dict,
    )

    @field_validator("plugins")
    @classmethod
    def _validate_plugin_names(cls, value: dict) -> dict:
        """Ensure plugin names are valid Python identifiers."""
        for name in value.keys():
            if not name.isidentifier():
                msg = f"Invalid plugin name '{name}'. Must be a valid Python identifier"
                raise ValueError(msg)
        return value

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> PluginManifest:
        """Load and validate manifest from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Validated PluginManifest instance

        Raises:
            FileNotFoundError: If yaml_path doesn't exist
        """
        if not yaml_path.exists():
            msg = f"Plugin manifest not found: {yaml_path}"
            raise FileNotFoundError(msg)

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return cls(plugins={})

        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes None values)."""
        return self.model_dump(exclude_none=True, mode="json")

    def to_yaml(self, yaml_path: Path) -> None:
        """Save manifest to YAML file.

        Args:
            yaml_path: Path where YAML file should be saved
        """
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(exclude_none=True),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
