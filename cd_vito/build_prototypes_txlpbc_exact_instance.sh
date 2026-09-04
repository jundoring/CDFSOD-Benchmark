#!/bin/bash
# Build CD-ViTO class prototypes for the TXL-PBC exact-K-instance policy
# (FT-FSOD counterpart at ../FT-FSOD/run_ftfsod_exactinstance_traineval.sh).
# Mirrors build_prototypes_txlpbc_hituav_seed1.sh but targets the
# datasets/register_txlpbc_exact_instance.py catalog names
# (txlpbc_exact_train_{K}shot_seed{N}) instead of the K-image
# txlpbc_train_{K}shot_seed1 names, and is TXL-PBC only (no HIT-UAV).
#
# GPU-bound. Do not run concurrently with another GPU job on this box.
#
# Usage: ./build_prototypes_txlpbc_exact_instance.sh [SEED]   (default SEED=1)
set -euo pipefail

SEED="${1:-1}"
mkdir -p prototypes_init

SHOTS=(1 5 10)
MODEL="vitl14"

for shot in "${SHOTS[@]}"; do
    name="txlpbc_exact_train_${shot}shot_seed${SEED}"
    echo "==== extract_instance_prototypes: ${name} (${MODEL}) ===="
    python3 ./tools/extract_instance_prototypes.py \
        --dataset "${name}" --out_dir prototypes_init --model "${MODEL}" \
        --epochs 1 --use_bbox yes --without_mask True

    echo "==== run_sinkhorn_cluster: ${name} (${MODEL}, num_prototypes=${shot}) ===="
    python3 ./tools/run_sinkhorn_cluster.py \
        --inp "prototypes_init/${name}.${MODEL}.bbox.pkl" \
        --epochs 30 --momentum 0.002 --num_prototypes "${shot}"

    out_pkl="prototypes_init/${name}.${MODEL}.bbox.p${shot}.sk.pkl"
    if [ ! -f "${out_pkl}" ]; then
        echo "FATAL: expected prototype file not found: ${out_pkl}"
        exit 1
    fi
    echo "OK: ${out_pkl}"
done

echo "All TXL-PBC exact-instance seed${SEED} prototypes built."
