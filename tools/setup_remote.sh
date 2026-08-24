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

uv sync
uv pip install -e .

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (Backblaze B2 key id) before running this script}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY (Backblaze B2 application key) before running this script}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"

repo_root="$PWD"
db_dir="$(uv run python -c 'import yaml; from pathlib import Path; print(Path(yaml.safe_load(open("configs/download.yaml"))["data_home"]).resolve())')"
datasets="$(uv run python -c 'import yaml; print(" ".join(yaml.safe_load(open("configs/download.yaml"))["datasets"]))')"

echo "Pulling data from musicality_db..."
if [ -d "$db_dir/.git" ]; then
    git -C "$db_dir" pull
else
    git clone https://github.com/luczeng/musicality_db.git "$db_dir"
fi
# Scoped to configs/download.yaml's dataset list (plus splits, needed
# regardless of which datasets are trained on) rather than a bare `dvc pull`
# — a fresh remote instance shouldn't have to pull every dataset in the repo.
(cd "$db_dir" && uv run --project "$repo_root" dvc pull splits $datasets)

echo "Logging in to Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Setup complete — ready to train."
