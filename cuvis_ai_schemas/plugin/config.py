"""Plugin configuration schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BasePluginConfig(BaseModel):
    """Base plugin configuration with strict validation.

    All plugin types inherit from this base class to ensure
    consistent validation and error handling.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields (catch typos)
        validate_assignment=True,  # Validate on attribute assignment
    )

    provides: list[str] = Field(
        description="List of fully-qualified class paths this plugin provides",
        min_length=1,  # At least one class required
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _BasePluginConfig:
        """Create from dictionary."""
        return cls.model_validate(data)

    @field_validator("provides")
    @classmethod
    def _validate_class_paths(cls, value: list[str]) -> list[str]:
        """Ensure class paths are well-formed."""
        for class_path in value:
            if not class_path or "." not in class_path:
                msg = (
                    f"Invalid class path '{class_path}'. "
                    "Must be fully-qualified (e.g., 'package.module.ClassName')"
                )
                raise ValueError(msg)
        return value


class GitPluginConfig(_BasePluginConfig):
    """Git repository plugin configuration.

    Supports:
    - SSH URLs: git@gitlab.com:user/repo.git
    - HTTPS URLs: https://github.com/user/repo.git
    - Git tags only: v1.2.3, v0.1.0-alpha, etc.

    Note: Branches and commit hashes are NOT supported for reproducibility.
    """

    repo: str = Field(
        description="Git repository URL (SSH or HTTPS)",
        min_length=1,
    )

    tag: str = Field(
        description="Git tag (e.g., v1.2.3, v0.1.0-alpha). "
        "Branches and commit hashes are not supported.",
        min_length=1,
    )

    @field_validator("repo")
    @classmethod
    def _validate_repo_url(cls, value: str) -> str:
        """Validate Git repository URL format."""
        if not (
            value.startswith("git@") or value.startswith("https://") or value.startswith("http://")
        ):
            msg = f"Invalid repo URL '{value}'. Must start with 'git@', 'https://', or 'http://'"
            raise ValueError(msg)
        return value

    @field_validator("tag")
    @classmethod
    def _validate_tag(cls, value: str) -> str:
        """Validate Git tag is not empty."""
        if not value.strip():
            msg = "Git tag cannot be empty"
            raise ValueError(msg)
        return value.strip()


class LocalPluginConfig(_BasePluginConfig):
    """Local filesystem plugin configuration.

    Supports:
    - Absolute paths: /home/user/my-plugin
    - Relative paths: ../my-plugin (resolved relative to manifest file)
    - Windows paths: C:\\Users\\user\\my-plugin
    """

    path: str = Field(
        description="Absolute or relative path to plugin directory",
        min_length=1,
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Validate path is not empty."""
        if not value.strip():
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return value.strip()

    def resolve_path(self, manifest_dir: Path) -> Path:
        """Resolve relative paths to absolute paths.

        Args:
            manifest_dir: Directory containing the manifest file

        Returns:
            Absolute path to plugin directory
        """
        plugin_path = Path(self.path)
        if not plugin_path.is_absolute():
            plugin_path = (manifest_dir / plugin_path).resolve()
        return plugin_path
