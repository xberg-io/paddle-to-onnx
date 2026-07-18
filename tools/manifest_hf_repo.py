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

"""Generate and upload a manifest.json for an already-hosted HuggingFace model repo.

Lists every ``.onnx`` (and ``dict.txt``) in the repo, downloads each, computes its
sha256, and records static ONNX info (opset, node count, input/output shapes) — the
same schema the export tooling emits — then writes a repo-root ``manifest.json`` so
consumers can pin exact checksums for the whole fleet. Introspection is static
(``onnx.load`` only, no inference), so no dummy inputs are needed.

Uploads are idempotent: only manifest.json is written, and only if it changed.

Usage:
    python tools/manifest_hf_repo.py --repo xberg-io/paddleocr-onnx-models
    python tools/manifest_hf_repo.py --repo xberg-io/layout-models --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import HfApi, get_token, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _shape(type_proto) -> list:
    """Return a declared tensor shape as a list of ints or dim-param strings."""
    return [dim.dim_value or dim.dim_param or "?" for dim in type_proto.tensor_type.shape.dim]


def onnx_static_info(path: Path) -> dict:
    """Read opset, node count, and declared I/O shapes without running inference."""
    model = onnx.load(str(path))
    return {
        "opset_version": model.opset_import[0].version,
        "nodes": len(model.graph.node),
        "inputs": [{"name": i.name, "shape": _shape(i.type)} for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": _shape(o.type)} for o in model.graph.output],
    }


def build_manifest(repo: str, files: list[str]) -> dict[str, dict]:
    """Download each model/dict file and assemble its manifest entry."""
    manifest: dict[str, dict] = {}
    for rel in sorted(files):
        local = Path(hf_hub_download(repo, rel))
        entry = {"sha256": sha256_of(local), "size_bytes": local.stat().st_size}
        if rel.endswith(".onnx"):
            entry.update(onnx_static_info(local))
            logger.info("Indexed %s (%d nodes, opset %d)", rel, entry["nodes"], entry["opset_version"])
        else:
            entry["num_chars"] = len(local.read_text(encoding="utf-8").splitlines())
            logger.info("Indexed %s (%d chars)", rel, entry["num_chars"])
        manifest[rel] = entry
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + upload a manifest.json for a hosted HF model repo")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. xberg-io/paddleocr-onnx-models")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env or `hf auth login`)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the manifest without uploading",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)

    files = api.list_repo_files(args.repo, repo_type="model")
    targets = [f for f in files if f.endswith(".onnx") or Path(f).name == "dict.txt"]
    if not targets:
        logger.error("No .onnx or dict.txt files found in %s", args.repo)
        sys.exit(1)

    manifest = build_manifest(args.repo, targets)
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    logger.info("Manifest covers %d files; manifest.json sha256=%s", len(manifest), manifest_sha)

    if args.dry_run:
        print(body)
        return

    if not token:
        logger.error("No HF token: pass --token, set HF_TOKEN, or run `hf auth login`")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        manifest_path.write_text(body)
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo="manifest.json",
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add manifest.json ({len(manifest)} files, sha256 {manifest_sha[:12]})",
        )
    logger.info("Uploaded manifest.json to https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
