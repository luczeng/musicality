#!/usr/bin/env bash
# Bootstrap a fresh remote instance (e.g. vast.ai) for training.
#
# Expects AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (Backblaze B2 credentials for the
# DVC remote) and WANDB_API_KEY to already be set in the environment — on vast.ai these
# come from the instance template's environment variables.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python 3.14..."
uv python install 3.14

uv sync
uv pip install -e .

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (Backblaze B2 key id) before running this script}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY (Backblaze B2 application key) before running this script}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"

echo "Pulling data from Backblaze..."
uv run dvc pull

echo "Logging in to Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Fetching mirdata indices for DVC-pulled datasets..."
uv run python - <<'EOF'
import yaml
from pathlib import Path

import mirdata

cfg = yaml.safe_load(Path("configs/download.yaml").read_text())
data_home = Path(cfg["data_home"])

for name in cfg["datasets"]:
    if not (data_home / f"{name}.dvc").exists():
        continue
    print(f"  {name}")
    ds = mirdata.initialize(name, data_home=str(data_home / name))
    ds.download(partial_download=["index"])
EOF

echo "Setup complete — ready to train."
