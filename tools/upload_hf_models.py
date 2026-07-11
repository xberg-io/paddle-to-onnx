#!/usr/bin/env python3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upload exported ONNX models to a distribution HuggingFace repo.

Reads the output directory produced by export_hf_models.py and publishes each
model that defines an ``upload_path`` using a versioned layout:

    <upload_path>/model.onnx          e.g. v6/det/medium/model.onnx
    <upload_path>/dict.txt            recognition models only (char dict)
    <generation>/manifest.json        per-generation manifest at the version root

The recognition ``dict.txt`` is extracted from the ONNX "character" metadata so it
always matches the embedded dictionary. Uploads are idempotent: upload_folder only
pushes changed files and never deletes existing repo content, so the hand-written
repo card and other generations are left untouched.

Usage:
    python tools/upload_hf_models.py --export-dir /tmp/exports --generation v6
    python tools/upload_hf_models.py --export-dir /tmp/exports --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import HfApi, get_token

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hf_models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO = "xberg-io/paddleocr-onnx-models"


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def character_metadata(onnx_path: Path) -> str | None:
    """Return the embedded "character" metadata value, or None if absent."""
    model = onnx.load(str(onnx_path))
    for prop in model.metadata_props:
        if prop.key == "character":
            return prop.value
    return None


def stage_model(
    name: str,
    upload_path: str,
    source_repo: str,
    export_dir: Path,
    staging_dir: Path,
    export_manifest: dict,
) -> dict[str, dict]:
    """Stage one model's files under staging_dir and return its manifest items.

    Returns a mapping of repo-relative path -> file metadata (empty if the
    exported ONNX is missing). Recognition models additionally emit dict.txt.
    """
    source = export_dir / f"{name}.onnx"
    if not source.exists():
        logger.warning("Skipping %s: %s not found in export dir", name, source.name)
        return {}

    dest_dir = staging_dir / upload_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx_dest = dest_dir / "model.onnx"
    shutil.copy2(source, onnx_dest)

    onnx_rel = f"{upload_path}/model.onnx"
    info = export_manifest.get(name, {})
    items: dict[str, dict] = {
        onnx_rel: {
            "sha256": info.get("sha256") or sha256_of(onnx_dest),
            "size_bytes": info.get("size_bytes", onnx_dest.stat().st_size),
            "opset_version": info.get("opset_version"),
            "nodes": info.get("nodes"),
            "inputs": info.get("inputs"),
            "outputs": info.get("outputs"),
            "source_repo": source_repo,
        }
    }

    character = character_metadata(onnx_dest)
    if character is not None:
        dict_dest = dest_dir / "dict.txt"
        dict_dest.write_text(character + "\n", encoding="utf-8")
        num_chars = len(character.split("\n"))
        items[f"{upload_path}/dict.txt"] = {
            "sha256": sha256_of(dict_dest),
            "size_bytes": dict_dest.stat().st_size,
            "num_chars": num_chars,
        }
        logger.info("Staged %s (+ dict.txt, %d chars)", onnx_rel, num_chars)
    else:
        logger.info("Staged %s", onnx_rel)

    return items


def write_generation_manifests(staging_dir: Path, items: dict[str, dict]) -> list[str]:
    """Write <generation>/manifest.json for each generation present. Returns the generations."""
    by_generation: dict[str, dict] = {}
    for rel, entry in items.items():
        by_generation.setdefault(rel.split("/")[0], {})[rel] = entry

    for generation, files in by_generation.items():
        manifest_path = staging_dir / generation / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
        logger.info("Wrote %s/manifest.json (%d files)", generation, len(files))

    return sorted(by_generation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload exported ONNX models to a HuggingFace repo")
    parser.add_argument("--export-dir", required=True, help="Directory with {name}.onnx + manifest.json")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target HF repo id")
    parser.add_argument("--generation", default=None, help='Only upload this generation prefix, e.g. "v6"')
    parser.add_argument("--token", default=None, help="HF token (defaults to the HF_TOKEN env var)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage locally and print the layout without uploading",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    export_manifest: dict = {}
    manifest_file = export_dir / "manifest.json"
    if manifest_file.exists():
        export_manifest = json.loads(manifest_file.read_text())

    selected = {
        name: config.upload_path
        for name, config in MODELS.items()
        if config.upload_path and (args.generation is None or config.upload_path.split("/")[0] == args.generation)
    }
    if not selected:
        logger.error("No models with upload_path match generation=%s", args.generation)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        items: dict[str, dict] = {}
        for name, upload_path in selected.items():
            items.update(stage_model(name, upload_path, MODELS[name].hf_repo, export_dir, staging, export_manifest))

        if not items:
            logger.error("Nothing staged — no exported .onnx files found in %s", export_dir)
            sys.exit(1)

        generations = write_generation_manifests(staging, items)

        if args.dry_run:
            logger.info("Dry run — staged tree for %s:", args.repo)
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    logger.info("  %s", path.relative_to(staging))
            return

        token = args.token or os.environ.get("HF_TOKEN") or get_token()
        if not token:
            logger.error("No HF token: pass --token, set HF_TOKEN, or run `hf auth login`")
            sys.exit(1)

        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=str(staging),
            commit_message=f"Upload {', '.join(generations)} ONNX models + manifest",
        )
        logger.info("Uploaded to https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
