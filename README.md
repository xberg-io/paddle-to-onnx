# Paddle2ONNX

Convert PaddlePaddle models (PIR format) to ONNX.

This is an internal fork of [PaddlePaddle/Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX), trimmed and modernized by Kreuzberg, Inc. **This fork is unsupported and not published to PyPI. Use it only with full awareness that we provide no guarantees.**

## Requirements

- Python 3.10+
- PaddlePaddle 3.3 or later (PIR-based)
- CMake 3.16+ (for building the C++ extension)

## Install

Clone the repo and build from source:

```bash
git clone https://github.com/xberg-io/paddle-to-onnx
cd paddle-to-onnx
task setup
```

This runs `uv sync` (installs dependencies) and builds the C++ extension via CMake.

## Quick Start

Convert a PaddlePaddle model (`.json` + `.pdiparams`) to ONNX:

```python
import paddle2onnx

paddle2onnx.export(
    model_filename="model.json",
    params_filename="model.pdiparams",
    save_file="model.onnx",
    opset_version=17,
)
```

## CLI

```bash
paddle2onnx \
  --model_dir /path/to/model \
  --model_filename model.json \
  --params_filename model.pdiparams \
  --save_file output.onnx \
  --opset_version 17
```

Run `paddle2onnx --help` for all options.

## Development

- `task build` — build the C++ extension
- `task test` — run pytest
- `task lint` — run poly linter
- `task format` — format code (poly)
- `task typecheck` — run pyrefly type checker
- `task export:<model>` — convert a model to ONNX (requires `tools/export_hf_models.py`)

## Changes from Upstream

- Modernized tooling: `uv` for packaging, `poly` for lint/format, `pyrefly` for type checking (dropped standalone `ruff`/`mypy`)
- PIR-based conversion targeting PaddlePaddle 3.3+ (ONNX opset 7–25); deprecated `.pdmodel` inputs are auto-translated to PIR
- Build fixes for current toolchains (protobuf 35, recent libc++/Xcode)

## License

Apache-2.0 (see LICENSE)
