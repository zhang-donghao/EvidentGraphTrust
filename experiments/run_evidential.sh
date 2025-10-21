#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/models.yaml \
  --head evidential \
  --log_dir runs/evidential_default
