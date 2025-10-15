#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data}
mkdir -p "${DATA_ROOT}"

echo "Please download VeReMi and TON_IoT datasets manually following the instructions in docs/dataset_preprocessing.md."
echo "Place the processed PyTorch Geometric tensors under \"${DATA_ROOT}/veremi\" and \"${DATA_ROOT}/toni_iot\" respectively."
