#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/models.yaml \
  --head mlp \
  --log_dir runs/baseline_mlp
