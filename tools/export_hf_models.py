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

"""Export PaddlePaddle inference models from HuggingFace to ONNX.

Downloads models from the PaddlePaddle HuggingFace organization, converts them
to ONNX using paddle2onnx, applies ORT compatibility fixes, and validates the
output with ONNX Runtime.

Usage:
    python tools/export_hf_models.py --model PaddlePaddle/SLANet_plus --output-dir /tmp/exports
    python tools/export_hf_models.py --all --output-dir /tmp/exports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from onnx import helper

import paddle2onnx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Suppress ORT warnings during validation (shape mismatch warnings are expected
# for models with Loop nodes and are benign after our fixes)
ort.set_default_logger_severity(3)


@dataclass
class ModelConfig:
    """Configuration for a PaddlePaddle model to export."""

    hf_repo: str
    model_filename: str = "inference.json"
    params_filename: str = "inference.pdiparams"
    opset_version: int = 17
    # Map of input_name -> (shape, dtype). If None, auto-detected from the model.
    inputs: dict[str, tuple[tuple[int, ...], str]] | None = None


# Models to export — add new models here
MODELS: dict[str, ModelConfig] = {
    # --- Text Detection (v5 latest) ---
    "PP-OCRv5_server_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_server_det",
    ),
    "PP-OCRv5_mobile_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_mobile_det",
    ),
    # --- Text Recognition (v5 latest) ---
    "PP-OCRv5_server_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_server_rec",
    ),
    "PP-OCRv5_mobile_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_mobile_rec",
    ),
    "en_PP-OCRv5_mobile_rec": ModelConfig(
        hf_repo="PaddlePaddle/en_PP-OCRv5_mobile_rec",
    ),
    # --- Document Layout Analysis ---
    "PP-DocLayoutV3": ModelConfig(
        hf_repo="PaddlePaddle/PP-DocLayoutV3",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 800, 800), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    # --- Table Structure Recognition ---
    "SLANet_plus": ModelConfig(
        hf_repo="PaddlePaddle/SLANet_plus",
    ),
    "SLANeXt_wired": ModelConfig(
        hf_repo="PaddlePaddle/SLANeXt_wired",
    ),
    "SLANeXt_wireless": ModelConfig(
        hf_repo="PaddlePaddle/SLANeXt_wireless",
    ),
    # --- Table Cell Detection ---
    "RT-DETR-L_wired_table_cell_det": ModelConfig(
        hf_repo="PaddlePaddle/RT-DETR-L_wired_table_cell_det",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 640, 640), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    "RT-DETR-L_wireless_table_cell_det": ModelConfig(
        hf_repo="PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 640, 640), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    # --- Document Orientation / Classification ---
    "PP-LCNet_x1_0_doc_ori": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_doc_ori",
    ),
    "PP-LCNet_x1_0_textline_ori": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_textline_ori",
    ),
    "PP-LCNet_x1_0_table_cls": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_table_cls",
    ),
}


def download_model(config: ModelConfig, target_dir: Path) -> Path:
    """Download model files from HuggingFace."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in [config.model_filename, config.params_filename]:
        hf_hub_download(config.hf_repo, fname, local_dir=str(target_dir))
        logger.info("Downloaded %s/%s", config.hf_repo, fname)
    return target_dir


def export_to_onnx(config: ModelConfig, model_dir: Path, output_path: Path) -> None:
    """Export a PaddlePaddle model to ONNX."""
    model_file = str(model_dir / config.model_filename)
    params_file = str(model_dir / config.params_filename)

    logger.info(
        "Exporting %s to ONNX (opset %d)...", config.hf_repo, config.opset_version
    )

    paddle2onnx.export(
        model_filename=model_file,
        params_filename=params_file,
        save_file=str(output_path),
        opset_version=config.opset_version,
        auto_upgrade_opset=False,
        verbose=False,
        enable_onnx_checker=True,
        enable_experimental_op=True,
        enable_optimize=True,
        deploy_backend="onnxruntime",
        calibration_file="",
        external_file="",
        export_fp16_model=False,
        optimize_tool="None",
    )
    logger.info("Exported to %s", output_path)


def _promote_scalar_to_1d(shape_proto, dtype: int) -> bool:
    """Change a scalar shape declaration to [1]. Returns True if changed."""
    dims = [d.dim_value or d.dim_param for d in shape_proto.dim]
    if dims == []:
        shape_proto.CopyFrom(
            helper.make_tensor_type_proto(dtype, [1]).tensor_type.shape
        )
        return True
    return False


def fix_ort_compatibility(model_path: Path) -> None:
    """Apply post-export fixes for ORT compatibility.

    Paddle2ONNX exports Loop nodes where the ONNX Loop spec mandates the
    iteration counter as a [1]-shaped tensor, but scalar ([]) shapes are used
    for loop-carried state, conditions, and If branch outputs. ORT's shape
    inference propagates [1] from the iteration counter through comparison and
    logical ops, then rejects the model when it hits a [] declaration.

    The fix: promote ALL scalar shapes to [1] in the Loop body, its If
    sub-branches, and the graph-level Constants that feed into the Loop.
    """
    model = onnx.load(str(model_path))
    fixed = False

    for node in model.graph.node:
        if node.op_type != "Loop":
            continue

        loop_inputs = set(node.input)

        # Fix graph-level Constant nodes that feed scalar values into the Loop
        for gnode in model.graph.node:
            if gnode.op_type == "Constant" and any(
                o in loop_inputs for o in gnode.output
            ):
                for a in gnode.attribute:
                    if a.name == "value" and list(a.t.dims) == []:
                        a.t.dims[:] = [1]
                        fixed = True

        # Fix graph-level initializers that feed into the Loop
        for init in model.graph.initializer:
            if init.name in loop_inputs and list(init.dims) == []:
                init.dims[:] = [1]
                fixed = True

        for attr in node.attribute:
            if attr.name != "body":
                continue
            body = attr.g

            # Fix all scalar body inputs, outputs, value_info
            for inp in body.input:
                fixed |= _promote_scalar_to_1d(
                    inp.type.tensor_type.shape, inp.type.tensor_type.elem_type
                )
            for out in body.output:
                fixed |= _promote_scalar_to_1d(
                    out.type.tensor_type.shape, out.type.tensor_type.elem_type
                )
            for vi in body.value_info:
                fixed |= _promote_scalar_to_1d(
                    vi.type.tensor_type.shape, vi.type.tensor_type.elem_type
                )

            # Fix scalar initializers and Constant nodes in body
            for init in body.initializer:
                if list(init.dims) == []:
                    init.dims[:] = [1]
                    fixed = True
            for n in body.node:
                if n.op_type == "Constant":
                    for a in n.attribute:
                        if a.name == "value" and list(a.t.dims) == []:
                            a.t.dims[:] = [1]
                            fixed = True

                # Fix If sub-branches inside the Loop body
                if n.op_type == "If":
                    for if_attr in n.attribute:
                        if if_attr.name in ("then_branch", "else_branch"):
                            branch = if_attr.g
                            for out in branch.output:
                                fixed |= _promote_scalar_to_1d(
                                    out.type.tensor_type.shape,
                                    out.type.tensor_type.elem_type,
                                )
                            for bn in branch.node:
                                if bn.op_type == "Constant":
                                    for a in bn.attribute:
                                        if a.name == "value" and list(a.t.dims) == []:
                                            a.t.dims[:] = [1]
                                            fixed = True
                            for init in branch.initializer:
                                if list(init.dims) == []:
                                    init.dims[:] = [1]
                                    fixed = True

    if fixed:
        logger.info("Applied ORT Loop/If scalar-to-[1] shape fix")
        onnx.save(model, str(model_path))


ONNX_DTYPE_TO_NP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    5: "int16",
    6: "int32",
    7: "int64",
    10: "float16",
    11: "float64",
}


def _build_dummy_inputs(
    config: ModelConfig, session: ort.InferenceSession
) -> dict[str, np.ndarray]:
    """Build dummy input tensors for validation, using config overrides or model metadata."""
    input_feed = {}
    for inp in session.get_inputs():
        if config.inputs and inp.name in config.inputs:
            shape, dtype = config.inputs[inp.name]
        else:
            # Auto-detect from model: replace dynamic dims with reasonable defaults
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
            dtype = inp.type.replace("tensor(", "").replace(")", "")
        # Map ORT type strings to numpy dtypes (np.float was removed in numpy 2.0)
        np_dtype = {
            "float": np.float32,
            "double": np.float64,
            "int64": np.int64,
            "int32": np.int32,
        }.get(dtype, getattr(np, dtype, np.float32))
        input_feed[inp.name] = np.random.randn(*shape).astype(np_dtype)
    return input_feed


def validate_onnx(config: ModelConfig, model_path: Path) -> dict:
    """Validate ONNX model with checker and ORT inference."""
    # ONNX checker
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    logger.info("ONNX checker: PASSED")

    # ORT inference
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_feed = _build_dummy_inputs(config, session)
    results = session.run(None, input_feed)

    outputs = []
    for i, r in enumerate(results):
        logger.info("Output %d: shape=%s dtype=%s", i, r.shape, r.dtype)
        outputs.append({"shape": list(r.shape), "dtype": str(r.dtype)})

    logger.info("ORT inference: PASSED")

    # Compute SHA256
    with open(model_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    size = model_path.stat().st_size

    return {
        "model": config.hf_repo,
        "opset_version": model.opset_import[0].version,
        "nodes": len(model.graph.node),
        "sha256": sha256,
        "size_bytes": size,
        "outputs": outputs,
        "inputs": [
            {
                "name": i.name,
                "shape": [
                    d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim
                ],
            }
            for i in model.graph.input
        ],
    }


def export_model(name: str, config: ModelConfig, output_dir: Path) -> dict | None:
    """Download, export, fix, and validate a single model."""
    logger.info("=" * 60)
    logger.info("Processing: %s (%s)", name, config.hf_repo)
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "paddle"
        download_model(config, model_dir)

        onnx_path = output_dir / f"{name}.onnx"
        export_to_onnx(config, model_dir, onnx_path)

    fix_ort_compatibility(onnx_path)

    info = validate_onnx(config, onnx_path)
    logger.info("SHA256: %s", info["sha256"])
    logger.info(
        "Size: %d bytes (%.2f MB)", info["size_bytes"], info["size_bytes"] / 1024 / 1024
    )

    return info


def _try_export(name: str, config: ModelConfig, output_dir: Path) -> dict | None:
    """Attempt to export a model, returning None on failure."""
    try:
        return export_model(name, config, output_dir)
    except Exception:
        logger.exception("Failed to export %s", name)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Export PaddlePaddle HF models to ONNX"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="Model name (e.g. SLANet_plus) or HF repo (e.g. PaddlePaddle/SLANet_plus)",
    )
    group.add_argument(
        "--all", action="store_true", help="Export all configured models"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to write JSON manifest of exported models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be exported without actually exporting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_export: dict[str, ModelConfig] = {}
    if args.all:
        models_to_export = MODELS
    else:
        # Support both "SLANet_plus" and "PaddlePaddle/SLANet_plus"
        model_key = args.model.split("/")[-1] if "/" in args.model else args.model
        if model_key not in MODELS:
            logger.error(
                "Unknown model: %s. Available: %s", model_key, list(MODELS.keys())
            )
            sys.exit(1)
        models_to_export = {model_key: MODELS[model_key]}

    if args.dry_run:
        logger.info("Dry run — would export %d models:", len(models_to_export))
        for name, config in models_to_export.items():
            logger.info(
                "  %s (%s, opset %d)", name, config.hf_repo, config.opset_version
            )
        return

    manifest = {}
    failed = []

    for name, config in models_to_export.items():
        info = _try_export(name, config, output_dir)
        if info is not None:
            manifest[name] = info
        else:
            failed.append(name)

    # Write manifest
    manifest_path = (
        Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    if failed:
        logger.error("Failed models: %s", failed)
        sys.exit(1)

    logger.info("All exports completed successfully")


if __name__ == "__main__":
    main()
