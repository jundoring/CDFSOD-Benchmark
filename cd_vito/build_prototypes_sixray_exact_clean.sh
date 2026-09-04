#!/bin/bash
# Build CD-ViTO class prototypes for the fair-conditions SIXray-D
# exact-K-instance seed1 re-run (canonical Dataset/SIXray-D root, via
# datasets/register_sixray_exact_instance.py's sixray_exact_train_{K}shot_seed1
# catalog names). GPU-bound - do not run concurrently with another GPU job.
set -euo pipefail

mkdir -p prototypes_init

SHOTS=(1 5 10)
MODEL="vitl14"

for shot in "${SHOTS[@]}"; do
    name="sixray_exact_train_${shot}shot_seed1"
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

echo "All SIXray-D exact-instance seed1 (clean) prototypes built."
