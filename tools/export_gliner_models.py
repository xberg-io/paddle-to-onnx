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

"""Export gliner-community GLiNER v2.5 models to span-mode ONNX.

Downloads the Apache-2.0 ``gliner-community`` v2.5 models, exports each to fp32
ONNX via the ``gliner`` library's built-in ``export_to_onnx`` (span mode
``markerV0``), and validates that the graph exposes the exact tensor interface
Xberg's ``ner-gliner`` backend requires. Writes the layout and ``checksums.sha256``
manifest consumed by the ``xberg-io/gliner-models`` HuggingFace repo.

Output layout under ``--output-dir``::

    models/gliner_<size>-v2.5/span/fp32/{model.onnx,tokenizer.json}
    checksums.sha256

Usage:
    python tools/export_gliner_models.py --all --output-dir /tmp/gliner
    python tools/export_gliner_models.py --model medium --output-dir /tmp/gliner --sanity

This tool has extra runtime deps not needed by the Paddle exporter — install
them into the same venv first::

    uv pip install gliner torch onnx

``torch``/``gliner`` currently need Python <= 3.12; create the venv with
``uv venv --python 3.12`` if the default interpreter is newer.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Source models — the Apache-2.0 gliner-community line (NOT the cc-by-nc-4.0
# urchade originals). Keys are the short size names accepted on the CLI.
MODELS: dict[str, str] = {
    "small": "gliner-community/gliner_small-v2.5",
    "medium": "gliner-community/gliner_medium-v2.5",
    "large": "gliner-community/gliner_large-v2.5",
}

OPSET = 19

# The span-mode (markerV0) tensor contract validated by xberg-gliner's
# session.rs. An export with any other names is rejected at load time, so we
# assert on it here rather than shipping a silently-incompatible model.
REQUIRED_INPUTS = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"}
REQUIRED_OUTPUTS = ["logits"]

SANITY_SENTENCE = "Barack Obama was born in Hawaii and worked at Microsoft."
SANITY_LABELS = ["person", "location", "organization"]


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(onnx_path: Path) -> None:
    """Assert the ONNX graph matches the GLiNER span-mode tensor contract."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = {i.name for i in model.graph.input}
    outputs = [o.name for o in model.graph.output]
    if inputs != REQUIRED_INPUTS:
        raise ValueError(f"input schema mismatch: got {sorted(inputs)}, want {sorted(REQUIRED_INPUTS)}")
    if outputs != REQUIRED_OUTPUTS:
        raise ValueError(f"output schema mismatch: got {outputs}, want {REQUIRED_OUTPUTS}")
    onnx.checker.check_model(str(onnx_path))
    logger.info("schema OK — inputs=%s outputs=%s", sorted(inputs), outputs)


def sanity_check(size: str, work_dir: Path) -> None:
    """Run real ONNX inference and assert the model detects a person entity.

    Loads from ``work_dir`` (holds ``gliner_config.json`` + tokenizer required by
    gliner's ONNX loader); its ``model.onnx`` is byte-identical to the deliverable.
    """
    from gliner import GLiNER  # local import: only needed with --sanity

    model = GLiNER.from_pretrained(str(work_dir), load_onnx_model=True, onnx_model_file="model.onnx")
    entities = model.predict_entities(SANITY_SENTENCE, SANITY_LABELS)
    found = [(e["text"], e["label"], round(float(e["score"]), 3)) for e in entities]
    logger.info("[%s] sanity entities = %s", size, found)
    if not any(e["label"] == "person" for e in entities):
        raise ValueError(f"[{size}] sanity check failed — no person entity: {found}")


def export_one(size: str, output_dir: Path, run_sanity: bool) -> dict:
    """Export a single model to ONNX, validate, and place the artifacts."""
    from gliner import GLiNER  # local import: only needed for actual export

    repo = MODELS[size]
    final_dir = output_dir / "models" / f"gliner_{size}-v2.5" / "span" / "fp32"
    final_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp) / size
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] loading %s", size, repo)
        model = GLiNER.from_pretrained(repo)
        span_mode = getattr(model.config, "span_mode", None)
        logger.info("[%s] class=%s span_mode=%s", size, type(model).__name__, span_mode)

        logger.info("[%s] exporting to ONNX (opset=%d, fp32)", size, OPSET)
        result = model.export_to_onnx(save_dir=str(work_dir), onnx_filename="model.onnx", quantize=False, opset=OPSET)
        onnx_path = Path(result["onnx_path"])
        validate_schema(onnx_path)
        if run_sanity:
            sanity_check(size, work_dir)

        shutil.copyfile(onnx_path, final_dir / "model.onnx")

    # tokenizer.json comes straight from the source repo (weights unmodified).
    tokenizer_src = hf_hub_download(repo, "tokenizer.json")
    shutil.copyfile(tokenizer_src, final_dir / "tokenizer.json")

    model_bytes = (final_dir / "model.onnx").stat().st_size
    logger.info("[%s] done — model.onnx %.2f MB", size, model_bytes / 1024 / 1024)
    return {"size": size, "model_bytes": model_bytes}


def write_checksums(output_dir: Path) -> Path:
    """Write checksums.sha256 (standard sha256sum format) over all artifacts."""
    files = sorted(p for p in output_dir.rglob("*") if p.name in ("model.onnx", "tokenizer.json") and p.is_file())
    lines = [f"{sha256_of(p)}  {p.relative_to(output_dir).as_posix()}" for p in files]
    checksums_path = output_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d entries)", checksums_path, len(lines))
    return checksums_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gliner-community v2.5 models to span-mode ONNX")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=sorted(MODELS), help="Export a single size")
    group.add_argument("--all", action="store_true", help="Export all sizes")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sanity", action="store_true", help="Run ONNX inference sanity check per model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(MODELS) if args.all else [args.model]

    for size in sizes:
        export_one(size, output_dir, args.sanity)
    write_checksums(output_dir)
    logger.info("All exports completed: %s", ", ".join(sizes))


if __name__ == "__main__":
    sys.exit(main())
