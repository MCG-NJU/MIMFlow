# Installation

## Environment

The recommended environment is:

- Python 3.10
- PyTorch with CUDA support
- Lightning 2.5.x

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version before or after installing this repository's Python dependencies.

## External Model Weights

MIMFlow uses `timm` and Hugging Face model loading for Transformer backbones and auxiliary feature targets. In offline environments, pre-download the relevant weights into the standard cache locations used by those libraries.

The cleaned configs avoid hard-coded private cache paths. Pass local paths through config overrides only when your environment requires them.

## Evaluation Environment

FID/IS/precision/recall are computed with the ADM guided-diffusion evaluator, not inside this training environment. Create the separate `adm-fid` environment described in `README.md` when scoring generated `samples.npz` files.

## Quick Import Check

```bash
python3 -m compileall main.py src
```
