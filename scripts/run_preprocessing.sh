#!/usr/bin/env bash
# Build the four-class protocol manifests and the preprocessed caches used in the paper.
#
# Usage:
#   cp configs/datasets.example.toml configs/datasets.local.toml   # then edit the two paths
#   bash scripts/run_preprocessing.sh [OUTPUT_ROOT]
#
# OUTPUT_ROOT defaults to ./outputs. The run is resumable: existing valid cache files are reused.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/outputs}"
PY="${PYTHON:-python}"
DATASETS="$ROOT/configs/datasets.local.toml"
PROTOCOL="$ROOT/configs/protocol_v1.toml"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$DATASETS" ] || { echo "Missing $DATASETS (copy configs/datasets.example.toml and set your paths)" >&2; exit 1; }

echo "[1/4] Protocol manifests -> $OUT/manifests"
"$PY" -m wifi_csi_quality.protocols --config "$DATASETS" --output-dir "$OUT/manifests"

cache() {  # $1 = manifest name, $2 = cache sub-folder
  echo "      cache: $1 -> $OUT/cache/$2"
  "$PY" -m wifi_csi_quality.preprocessing --dataset-config "$DATASETS" --protocol-config "$PROTOCOL" \
        --manifest "$OUT/manifests/$1.jsonl" --output-dir "$OUT/cache/$2"
}
echo "[2/4] Core cache (CSI-Bench in-distribution splits and XRF55 external test)"
for m in csi_bench_train_id csi_bench_val_id csi_bench_test_id xrf55_external_test; do cache "$m" core; done
echo "[3/4] Out-of-distribution cache (CSI-Bench official OOD splits)"
for m in csi_bench_test_cross_user csi_bench_test_cross_env csi_bench_test_cross_device csi_bench_test_cross_user_env; do cache "$m" ood; done
echo "[4/4] Cache audit"
"$PY" -m wifi_csi_quality.cache_audit --cache-root "$OUT/cache/core" --output-dir "$OUT/audit"
echo "Done. Caches in $OUT/cache/{core,ood}"
