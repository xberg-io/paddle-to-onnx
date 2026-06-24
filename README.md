# Paddle2ONNX — ONNX Model Export Pipeline

This is a detached fork of [PaddlePaddle/Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX), maintained by [kreuzberg-dev](https://github.com/xberg-io) to provide automated ONNX model exports of PaddleOCR inference models for use with ONNX Runtime.

## Purpose

PaddlePaddle publishes inference models in their native format (`.json` + `.pdiparams`). Most ML runtimes outside the Paddle ecosystem (ONNX Runtime, TensorRT, OpenVINO) require ONNX format. This repo automates the conversion pipeline:

1. Downloads latest-gen PaddleOCR models from [HuggingFace](https://huggingface.co/PaddlePaddle)
2. Exports them to ONNX using Paddle2ONNX (opset 17)
3. Applies ORT compatibility fixes (Loop node shape inference workarounds)
4. Validates every model with the ONNX checker and ONNX Runtime inference
5. Publishes validated `.onnx` files as GitHub release assets

## Exported Models

All models are Apache 2.0 licensed, exported with ONNX opset 17.

### Text Detection
| Model | Size | Description |
|-------|------|-------------|
| `PP-OCRv5_server_det` | 84 MB | High-accuracy text region detection. Outputs bounding polygons around text areas. For server-side batch processing. |
| `PP-OCRv5_mobile_det` | 4.5 MB | Lightweight text detection. Same function, optimized for speed and edge deployment. |

### Text Recognition
| Model | Size | Description |
|-------|------|-------------|
| `PP-OCRv5_server_rec` | 81 MB | Multilingual text recognition (Chinese, English, Japanese, Traditional Chinese). Reads characters from cropped text regions. |
| `PP-OCRv5_mobile_rec` | 16 MB | Multilingual recognition, smaller model for mobile/edge. |
| `en_PP-OCRv5_mobile_rec` | 7.5 MB | English/Latin-only recognition. Smallest and fastest for English workloads. |

### Document Layout Analysis
| Model | Size | Description |
|-------|------|-------------|
| `PP-DocLayoutV3` | 126 MB | Classifies document regions into 23 categories (title, paragraph, table, figure, header, footer, etc.). Used to understand page structure before content extraction. |

### Table Structure Recognition
| Model | Size | Description |
|-------|------|-------------|
| `SLANet_plus` | 7.4 MB | General-purpose table structure recognition. Outputs HTML structure tokens and cell bounding boxes. Good balance of speed and accuracy. |
| `SLANeXt_wired` | 348 MB | Optimized for tables with visible grid lines/borders. Higher accuracy on bordered tables. |
| `SLANeXt_wireless` | 348 MB | Optimized for borderless tables (whitespace-separated columns). Higher accuracy on unbordered tables. |

### Table Cell Detection
| Model | Size | Description |
|-------|------|-------------|
| `RT-DETR-L_wired_table_cell_det` | 123 MB | Object detection for individual cell bounding boxes in bordered tables. |
| `RT-DETR-L_wireless_table_cell_det` | 123 MB | Cell bounding box detection for borderless tables. |

### Utility Classifiers
| Model | Size | Description |
|-------|------|-------------|
| `PP-LCNet_x1_0_doc_ori` | 6.5 MB | Document orientation classifier (0/90/180/270 degrees). Run before OCR to auto-rotate pages. |
| `PP-LCNet_x1_0_textline_ori` | 6.5 MB | Text line orientation classifier. For documents with mixed text directions. |
| `PP-LCNet_x1_0_table_cls` | 6.5 MB | Wired vs. wireless table classifier. Used to select the appropriate SLANeXt or RT-DETR variant. |

### Typical OCR Pipeline

```
doc_ori → layout analysis → ┬─ text regions:  det → rec
                             └─ table regions: table_cls → SLANeXt + cell_det → rec
```

## Usage

### Download from Releases

Pre-built ONNX models are available as [GitHub release assets](https://github.com/xberg-io/paddle-to-onnx/releases). Each release includes a `manifest.json` with SHA256 checksums and model metadata.

### Export Locally

```bash
# Install dependencies
pip install paddlepaddle huggingface-hub onnx onnxruntime numpy
pip install -e .

# Export a single model
python tools/export_hf_models.py --model SLANet_plus --output-dir ./exports

# Export all models
python tools/export_hf_models.py --all --output-dir ./exports

# Dry run (list models without exporting)
python tools/export_hf_models.py --all --output-dir ./exports --dry-run
```

### CI/CD

The export pipeline runs automatically on release creation and uploads all `.onnx` files as release assets. It can also be triggered manually via workflow dispatch.

## Upstream

This fork tracks [PaddlePaddle/Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) (`develop` branch) with the following additions:

- `tools/export_hf_models.py` — automated export pipeline with ORT compatibility fixes
- `.github/workflows/export_models.yml` — CI workflow for model export and release publishing
- CMake and CI fixes for building on modern toolchains (Ubuntu 24.04, Python 3.13)

## License

[Apache-2.0](LICENSE)
