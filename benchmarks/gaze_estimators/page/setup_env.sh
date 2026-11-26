#!/usr/bin/env bash
# PaGE (arXiv 2607.04860) inference environment: Python 3.12, torch 2.6 cu124,
# transformers 5.6.2 (the released checkpoints need exactly this major line).
set -euo pipefail
ENVS=${ENVS:-$PWD/gaze_bench_envs}
export UV_PYTHON_INSTALL_DIR=$ENVS/_python UV_CACHE_DIR=$ENVS/_uv_cache
UV=${UV:-uv}
ENV=$ENVS/page
$UV python install 3.12
$UV venv --python 3.12 "$ENV"
$UV pip install --python "$ENV/bin/python" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
$UV pip install --python "$ENV/bin/python" "transformers==5.6.2" "timm>=0.9.0" "pillow>=10" "numpy>=1.24" "scipy>=1.10" "pandas>=2" tqdm safetensors "huggingface-hub>=0.16" opencv-python-headless
"$ENV/bin/python" -c "import torch, torchvision, transformers, timm; print('page env OK', torch.__version__, torchvision.__version__, transformers.__version__, timm.__version__, torch.cuda.is_available())"
