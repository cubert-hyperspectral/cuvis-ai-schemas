"""Tests for the manifest ``weights:`` block (:class:`PluginWeightEntry`, :class:`AuxFile`)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import (
    AuxFile,
    GitPluginSource,
    LocalPluginSource,
    PluginWeightEntry,
    load_plugin_manifest,
    parse_plugin_manifest,
    write_plugin_manifest,
)

REV = "6d25af14a085ff9d3e1342c35bae7c87de4811f4"
SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
AUX_SHA = "4616385e" + "0" * 56


def _entry(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "sam3",
        "display_name": "SAM3",
        "summary": "Magic wand, propagation, text prompts",
        "used_for": ["Point expansion", "Propagation"],
        "repo_id": "cubert-gmbh/sam3",
        "filename": "sam3.pt",
        "revision": REV,
        "sha256": SHA,
        "size_bytes": 3_400_000_000,
        "license": "SAM License",
        "license_file": "LICENSE",
        "explicit_path_hparams": ["checkpoint_path"],
    }
    data.update(overrides)
    return data


def _manifest(weights: list[dict[str, Any]]) -> GitPluginSource:
    return GitPluginSource.model_validate(
        {
            "name": "sam3",
            "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git",
            "tag": "v0.5.0",
            "capabilities": [{"class_name": "cuvis_ai_sam3.node.Sam3PointExpansion"}],
            "weights": weights,
        }
    )


def test_weight_entry_defaults_and_round_trip():
    """A minimal entry fills the optional fields and survives to_dict → from_dict."""
    entry = PluginWeightEntry(**_entry())
    assert entry.kind == "weights"
    assert entry.aux_files == []
    assert entry.aliases == []
    assert entry.selected_by is None
    assert entry.default is False
    assert entry.description == ""
    assert PluginWeightEntry.from_dict(entry.to_dict()) == entry


def test_manifest_without_weights_still_validates():
    """`weights` is optional: existing manifests load unchanged with an empty list."""
    manifest = parse_plugin_manifest(
        {
            "name": "sam3",
            "repo": "https://github.com/user/repo.git",
            "tag": "v1.0.0",
            "capabilities": [{"class_name": "pkg.mod.Node"}],
        }
    )
    assert manifest.weights == []


def test_manifest_with_weights_round_trips_through_dict():
    """A manifest carrying weights survives parse → to_dict → parse."""
    variant = _entry(
        name="efficienttam_s",
        display_name="RTSAM (EfficientTAM small)",
        summary="Point expansion, propagation (fastest RTSAM variant)",
        repo_id="cubert-gmbh/efficient-track-anything",
        filename="efficienttam_s.pt",
        license="Apache-2.0",
        aliases=["efficienttam"],
        selected_by="model_type",
        default=True,
        explicit_path_hparams=["model_dir"],
    )
    manifest = _manifest([_entry(), variant])
    restored = parse_plugin_manifest(manifest.to_dict())
    assert restored == manifest
    assert [w.name for w in restored.weights] == ["sam3", "efficienttam_s"]
    assert restored.weights[1].aliases == ["efficienttam"]
    assert restored.weights[1].default is True


def test_unknown_key_in_weight_entry_rejected():
    """extra='forbid' holds for the new models too."""
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(bogus="nope"))
    with pytest.raises(ValidationError):
        AuxFile(path="config.json", size_bytes=1, sha256=AUX_SHA, bogus=1)


def test_duplicate_weight_name_rejected():
    """Two rows with one name fail the manifest, naming the key."""
    with pytest.raises(ValueError, match="one namespace"):
        _manifest([_entry(), _entry(display_name="SAM3 again")])


def test_alias_colliding_with_another_row_rejected():
    """An alias that equals another row's name or alias is a conflict."""
    with pytest.raises(ValueError, match="'sam3' is declared by both"):
        _manifest(
            [
                _entry(name="efficienttam_s", aliases=["sam3"], selected_by="model_type"),
                _entry(),
            ]
        )
    with pytest.raises(ValueError, match="'fast' is declared by both"):
        _manifest(
            [
                _entry(name="a", aliases=["fast"], selected_by="model_type"),
                _entry(name="b", aliases=["fast"], selected_by="model_type"),
            ]
        )


def test_alias_equal_to_own_name_rejected():
    """A row may not alias itself, and aliases within a row are unique."""
    with pytest.raises(ValueError, match="repeats the weight's own name"):
        PluginWeightEntry(**_entry(aliases=["sam3"], selected_by="model_type"))
    with pytest.raises(ValueError, match="aliases must be unique"):
        PluginWeightEntry(**_entry(aliases=["x", "x"], selected_by="model_type"))


def test_names_and_aliases_are_registry_keys():
    """Whitespace, slashes and leading punctuation are rejected; dots and dashes pass."""
    for bad in ("", " sam3", "sam 3", "sam/3", "_sam3", "-sam3"):
        with pytest.raises(ValidationError):
            PluginWeightEntry(**_entry(name=bad))
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(aliases=["sam 3"], selected_by="model_type"))
    assert PluginWeightEntry(**_entry(name="sam2.1_hiera-large")).name == "sam2.1_hiera-large"


def test_hex_pins_validated():
    """revision is 40 and sha256 is 64 lowercase hex digits; nothing else passes."""
    with pytest.raises(ValueError, match="revision must be 40 lowercase hex"):
        PluginWeightEntry(**_entry(revision=REV[:-1]))
    with pytest.raises(ValueError, match="revision must be 40 lowercase hex"):
        PluginWeightEntry(**_entry(revision=REV.upper()))
    with pytest.raises(ValueError, match="sha256 must be 64 lowercase hex"):
        PluginWeightEntry(**_entry(sha256=SHA + "0"))
    with pytest.raises(ValueError, match="sha256 must be 64 lowercase hex"):
        AuxFile(path="config.json", size_bytes=1, sha256="abc")


def test_size_bytes_must_be_positive():
    """A zero-byte pin is a mistake, for the primary file and for aux files alike."""
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(size_bytes=0))
    with pytest.raises(ValidationError):
        AuxFile(path="config.json", size_bytes=0, sha256=AUX_SHA)


def test_summary_is_at_most_60_characters():
    """The summary is the row's sub-line; longer text belongs in description."""
    assert PluginWeightEntry(**_entry(summary="x" * 60)).summary == "x" * 60
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(summary="x" * 61))


def test_used_for_labels_non_empty_and_unique():
    """At least one label; no blanks, no repeats; labels are stripped."""
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(used_for=[]))
    with pytest.raises(ValueError, match="cannot be empty"):
        PluginWeightEntry(**_entry(used_for=["Backbone", "  "]))
    with pytest.raises(ValueError, match="must be unique"):
        PluginWeightEntry(**_entry(used_for=["Backbone", "Backbone"]))
    assert PluginWeightEntry(**_entry(used_for=[" Backbone "])).used_for == ["Backbone"]


def test_repo_id_and_paths_have_the_mirror_shape():
    """repo_id is 'owner/name'; filename and aux paths are relative in-repo paths."""
    with pytest.raises(ValueError, match="owner/name"):
        PluginWeightEntry(**_entry(repo_id="sam3"))
    with pytest.raises(ValueError, match="owner/name"):
        PluginWeightEntry(**_entry(repo_id="cubert-gmbh/sam3/extra"))
    for bad in ("/sam3.pt", "a/../sam3.pt", "a//sam3.pt", "./sam3.pt", "a\\sam3.pt"):
        with pytest.raises(ValueError, match="filename must"):
            PluginWeightEntry(**_entry(filename=bad))
    nested = PluginWeightEntry(**_entry(filename="dinomaly_cir_full_pipeline/dinomaly_cir.pt"))
    assert nested.filename == "dinomaly_cir_full_pipeline/dinomaly_cir.pt"
    with pytest.raises(ValueError, match=r"aux_files\[\]\.path must"):
        AuxFile(path="../config.json", size_bytes=1, sha256=AUX_SHA)


def test_license_file_is_nullable_and_a_bare_filename():
    """None means 'upstream states no weights licence'; a path is not a filename."""
    assert PluginWeightEntry(**_entry(license_file=None)).license_file is None
    assert PluginWeightEntry(**_entry(license_file=" LICENSE ")).license_file == "LICENSE"
    with pytest.raises(ValueError, match="bare filename"):
        PluginWeightEntry(**_entry(license_file="licenses/LICENSE"))
    with pytest.raises(ValueError, match="cannot be empty"):
        PluginWeightEntry(**_entry(license_file="  "))
    with pytest.raises(ValidationError):
        PluginWeightEntry(**_entry(license=""))


def test_default_requires_selected_by():
    """`default` names the row a pipeline gets without the selector, so it needs one."""
    with pytest.raises(ValueError, match="default=True requires 'selected_by'"):
        PluginWeightEntry(**_entry(default=True))
    entry = PluginWeightEntry(**_entry(default=True, selected_by="model_type"))
    assert entry.default is True and entry.selected_by == "model_type"


def test_selector_and_explicit_path_hparams_are_identifiers():
    """They name node hyper-parameters, so they must be Python identifiers, unique."""
    with pytest.raises(ValueError, match="selected_by must be a hyper-parameter name"):
        PluginWeightEntry(**_entry(selected_by="model-type"))
    with pytest.raises(ValueError, match="must be Python identifiers"):
        PluginWeightEntry(**_entry(explicit_path_hparams=["1path"]))
    with pytest.raises(ValueError, match="must be unique"):
        PluginWeightEntry(**_entry(explicit_path_hparams=["checkpoint_path", "checkpoint_path"]))


def test_trained_pipeline_rows_carry_aux_yaml_and_no_selector():
    """A trained pipeline pins its .pt and yaml; it is never picked by a hyper-parameter."""
    row = _entry(
        name="dinomaly_lentils_cir",
        display_name="Dinomaly lentils (CIR)",
        summary="Trained anomaly detector for the lentils demo",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id="cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils",
        filename="dinomaly_cir_full_pipeline/dinomaly_cir.pt",
        size_bytes=592_005_300,
        aux_files=[
            {
                "path": "dinomaly_cir_full_pipeline/dinomaly_cir.yaml",
                "size_bytes": 2744,
                "sha256": AUX_SHA,
            }
        ],
        license="Apache-2.0",
        license_file=None,
        explicit_path_hparams=[],
    )
    entry = PluginWeightEntry(**row)
    assert entry.kind == "trained_pipeline"
    assert entry.aux_files[0].path.endswith(".yaml")
    with pytest.raises(ValueError, match="never selected by a hyper-parameter"):
        PluginWeightEntry(**{**row, "selected_by": "model_type"})
    with pytest.raises(ValueError, match="never selected by a hyper-parameter"):
        PluginWeightEntry(**{**row, "default": True})


def test_aux_paths_unique_and_distinct_from_primary_file():
    """Two aux entries for one path, or an aux entry naming the primary file, are errors."""
    aux = {"path": "config.json", "size_bytes": 1, "sha256": AUX_SHA}
    with pytest.raises(ValueError, match="aux_files paths must be unique"):
        PluginWeightEntry(**_entry(aux_files=[aux, aux]))
    with pytest.raises(ValueError, match="must not repeat the primary file"):
        PluginWeightEntry(**_entry(aux_files=[{**aux, "path": "sam3.pt"}]))


def test_weight_entries_are_frozen():
    """Like capability entries, a weight entry is immutable once validated."""
    entry = PluginWeightEntry(**_entry())
    with pytest.raises(ValidationError):
        entry.name = "other"  # type: ignore[misc]


def test_write_keeps_weights_after_capabilities_and_omits_an_empty_block(tmp_path):
    """On disk `weights:` follows `capabilities:`; a manifest without weights has no key."""
    manifest = _manifest([_entry()])
    path = tmp_path / "sam3.yaml"
    write_plugin_manifest(manifest, path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = list(raw)
    assert keys.index("weights") == keys.index("capabilities") + 1
    assert raw["weights"][0]["license_file"] == "LICENSE"
    assert "selected_by" not in raw["weights"][0]  # None is dropped like package_name
    assert load_plugin_manifest(path) == manifest

    plain = LocalPluginSource(
        name="local", path=str(tmp_path), capabilities=[{"class_name": "pkg.mod.Node"}]
    )
    plain_path = tmp_path / "local.yaml"
    write_plugin_manifest(plain, plain_path)
    assert "weights" not in yaml.safe_load(plain_path.read_text(encoding="utf-8"))
    assert load_plugin_manifest(plain_path).weights == []
