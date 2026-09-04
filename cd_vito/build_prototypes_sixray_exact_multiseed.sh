#!/bin/bash
# Build CD-ViTO class prototypes for SIXray-D exact-K-instance seeds 2..10
# (seed1 prototypes already exist under the same naming from earlier work).
# GPU-bound. Do not run concurrently with another GPU job.
#
# Usage: ./build_prototypes_sixray_exact_multiseed.sh [SEED...]  (default: 2..10)
set -euo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(2 3 4 5 6 7 8 9 10)
fi

mkdir -p prototypes_init
SHOTS=(1 5 10)
MODEL="vitl14"

for seed in "${SEEDS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        name="sixray_exact_train_${shot}shot_seed${seed}"
        out_pkl="prototypes_init/${name}.${MODEL}.bbox.p${shot}.sk.pkl"
        if [ -f "${out_pkl}" ]; then
            echo "SKIP (already exists): ${out_pkl}"
            continue
        fi
        echo "==== extract_instance_prototypes: ${name} (${MODEL}) ===="
        python3 ./tools/extract_instance_prototypes.py \
            --dataset "${name}" --out_dir prototypes_init --model "${MODEL}" \
            --epochs 1 --use_bbox yes --without_mask True

        echo "==== run_sinkhorn_cluster: ${name} (${MODEL}, num_prototypes=${shot}) ===="
        python3 ./tools/run_sinkhorn_cluster.py \
            --inp "prototypes_init/${name}.${MODEL}.bbox.pkl" \
            --epochs 30 --momentum 0.002 --num_prototypes "${shot}"

        if [ ! -f "${out_pkl}" ]; then
            echo "FATAL: expected prototype file not found: ${out_pkl}"
            exit 1
        fi
        echo "OK: ${out_pkl}"
    done
done

echo "All SIXray-D exact-instance multiseed prototypes built."
