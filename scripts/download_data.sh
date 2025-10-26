#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data}
mkdir -p "${DATA_ROOT}"

echo "Please download VeReMi and TON_IoT datasets manually following the instructions in docs/dataset_preprocessing.md."
echo "After placing the raw CSV files locally, run:" \
    "python scripts/preprocess_veremi.py --raw-root <VeReMi_csv_dir> --output-root ${DATA_ROOT}" \
    "python scripts/preprocess_toniot.py --raw-root <TON_IoT_dir> --output-root ${DATA_ROOT}"
echo "Processed tensors will be saved under \"${DATA_ROOT}/veremi/processed\" and \"${DATA_ROOT}/toni_iot/processed\"."
