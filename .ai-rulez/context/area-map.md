---
type: Concept
title: Area Map
x-ai-rulez:
  kind: context
  id: area-map
  metadata:
    priority: high
---

## Area Map

Internal fork of Paddle2ONNX: converts PaddlePaddle (PIR) models to ONNX through a C++ core exposed to Python.

- `paddle2onnx/*.py` — Python package and `paddle2onnx` CLI (`command.py`, `convert.py`, `optimize.py`)
- `paddle2onnx/converter.*`, `cpp2py_export.cc` — C++ entry point and pybind bridge
- `paddle2onnx/parser/` — PIR and legacy program readers
- `paddle2onnx/mapper/` — per-op Paddle-to-ONNX mappers (`activation/`, `nn/`, `tensor/`, `detection/`, `quantize/`); new ops register via `register_mapper.h`
- `paddle2onnx/optimizer/` — ONNX graph fusion passes and fp16 conversion
- `paddle2onnx/proto/` — protobuf schema; `patch/paddle/` — patched Paddle headers
- `CMakeLists.txt`, `cmake/`, `setup.py`, `pyproject.toml` — native build (`mise run build`, `uv pip install -e .`); tests import the installed extension, so rerun it after any `*.cc`/`*.h` change before `mise run test`
- `tests/` — pytest op-level conversion tests (`mise run test`); match the upstream style of the file you touch (many are `OPConvertAutoScanTest` classes), not the python rule's function-based default
- `tools/export_hf_models.py`, `tools/upload_hf_models.py` — model export and HF upload (`mise run export <model>`; `mise run upload` publishes to Hugging Face with `HF_TOKEN`, run only when asked)
- `tools/tract_check/` — Rust loader that checks exported models resolve under tract-onnx
- `tools/onnx/`, `tools/paddle/`, `debug/` — model inspection and pruning utilities
- `exports/` — reference overlays and outputs for exported models
