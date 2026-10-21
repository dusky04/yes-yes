#!/usr/bin/env bash
# One-time setup on a fresh jarvislabs.ai GPU instance.
# Builds the project's uv environment (pyproject.toml + uv.lock, Python 3.13)
# and downloads + unzips the CricketEC clips.
#
#   bash benchmark/setup_jarvis.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CRICKET_EC_URL="https://drive.google.com/file/d/1QRM360a5HvRKvPF3k7vOT1PS0suxioJd/view?usp=sharing"

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> Project environment (uv sync)"
# --frozen: install exactly what uv.lock pins, never re-resolve on the server
uv sync --frozen
uv run --frozen python - <<'EOF'
import torch, torchvision, decord, sklearn
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | decord {decord.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT AVAILABLE: check the instance has a GPU and an up-to-date driver (nvidia-smi)")
print("CUDA:", torch.cuda.get_device_name(0))
EOF

echo "==> Dataset"
if [ -d CricketEC/train ] && [ -d CricketEC/val ] && [ -d CricketEC/test ]; then
    echo "CricketEC/ already present, skipping download"
else
    mkdir -p zipped_data
    [ -f zipped_data/CricketEC.zip ] || uv run --frozen gdown --fuzzy "$CRICKET_EC_URL" -O zipped_data/CricketEC.zip
    uv run --frozen python -m zipfile -e zipped_data/CricketEC.zip .
fi
for s in train val test; do
    printf "  %-5s %5s clips, %s classes\n" "$s" \
        "$(find "CricketEC/$s" -name '*.avi' | wc -l)" "$(find "CricketEC/$s" -mindepth 1 -maxdepth 1 -type d | wc -l)"
done

echo "==> Pre-downloading ImageNet weights"
uv run --frozen python - <<'EOF'
from torchvision.models import resnet34, efficientnet_b0, ResNet34_Weights, EfficientNet_B0_Weights
resnet34(weights=ResNet34_Weights.IMAGENET1K_V1); efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
EOF

echo "Setup done. Start the runs with:  bash benchmark/run_all.sh"
