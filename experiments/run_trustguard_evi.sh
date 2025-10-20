#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/models.yaml \
  --model trustguard \
  --head evidential \
  --dataset bitcoin_otc \
  --snapshots 10 \
  --eval_protocol single_observed \
  --log_dir runs/evi_trustguard_so
