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
split_name="$(uv run python -c 'import yaml; from musicality.loaders.beat_dataset import beat_split_name; c = yaml.safe_load(open("configs/beat_train.yaml")); print(beat_split_name(c["data"]["input"], c.get("binary_only", False)))')"
# Named by dataformat.yaml so a rename there doesn't leave this pulling a dead path.
splits_name="$(uv run python -c 'import musicality.dataformats as d; from pathlib import Path; print(Path(d.FORMAT.splits_dir).name)')"

echo "Pulling data from musicality_db..."
if [ -d "$db_dir/.git" ]; then
    git -C "$db_dir" pull
else
    git clone https://github.com/luczeng/musicality_db.git "$db_dir"
fi
# --local (not committed) so this only affects this instance: symlinking cache
# objects into the checkout avoids DVC's default copy, which otherwise doubles
# disk usage (full copy in .dvc/cache plus full copy in the checked-out dirs).
(cd "$db_dir" && uv run --project "$repo_root" dvc config --local cache.type symlink)
# Splits first: they list `<corpus>/<track_id>`, so they say what else to pull.
# Not configs/download.yaml's list — that one is what *mirdata* can fetch, and
# gtzan/rwc_genre reach the data repo by migration, so it is short by those two.
(cd "$db_dir" && uv run --project "$repo_root" dvc pull "$splits_name")

split_dir="$db_dir/$splits_name/$split_name"
corpora="$(cut -d/ -f1 "$split_dir/train.txt" "$split_dir/val.txt" | sort -u | tr '\n' ' ')"
echo "Split $split_name needs: $corpora"
(cd "$db_dir" && uv run --project "$repo_root" dvc pull $corpora)
echo "Logging in to Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Setup complete — ready to train."
