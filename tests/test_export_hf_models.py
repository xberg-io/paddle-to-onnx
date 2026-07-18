"""Unit tests for the HuggingFace export tooling character-dict resolution.

Covers the PP-OCRv6 change: v6 base repos have no config.json, so the recognition
character dict must be read from inference.yml, while v5's config.json still wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

export_hf_models = pytest.importorskip("export_hf_models")


def test_load_character_dict_prefers_config_json_when_present(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"PostProcess": {"character_dict": ["a", "b"]}}))
    (tmp_path / "inference.yml").write_text("PostProcess:\n  character_dict:\n  - x\n  - y\n")

    assert export_hf_models.load_character_dict(tmp_path) == ["a", "b"]


def test_load_character_dict_falls_back_to_inference_yml_for_v6(tmp_path: Path) -> None:
    # PP-OCRv6 base repos ship no config.json; the dict lives in inference.yml.
    (tmp_path / "inference.yml").write_text("PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - a\n")

    assert export_hf_models.load_character_dict(tmp_path) == ["!", "a"]


def test_load_character_dict_returns_none_without_a_dict(tmp_path: Path) -> None:
    # Detection/layout models carry no character dict (DBPostProcess, etc.).
    (tmp_path / "inference.yml").write_text("PostProcess:\n  name: DBPostProcess\n")

    assert export_hf_models.load_character_dict(tmp_path) is None


def test_load_character_dict_returns_none_when_no_metadata_files(tmp_path: Path) -> None:
    assert export_hf_models.load_character_dict(tmp_path) is None
