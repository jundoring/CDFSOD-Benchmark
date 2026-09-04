#!/bin/bash
# CD-ViTO HIT-UAV exact-K-instance queue, seeds 2..10 x 1/5/10-shot (27 jobs
# by default), using the new hituav_exact_train_{K}shot_seed{N} registration.
# seed1 is intentionally excluded by default - the existing seed1 result
# under the old hituav_train_{K}shot_seed1 name is reused as-is.
#
# PREREQUISITE: build_prototypes_hituav_exact_multiseed.sh must have been run
# first for the same seeds.
#
# Usage: ./run_cdvito_hituav_exact_multiseed.sh [SEED...]  (default: 2..10)
set -uo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(2 3 4 5 6 7 8 9 10)
fi

RESULT_DIR="output/vitl_hituav_exact_multiseed"
CKPT="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"
SHOTS=(1 5 10)

mkdir -p "${RESULT_DIR}"
QUEUE_LOG="${RESULT_DIR}/queue.log"
echo "==== Queue started at $(date -Iseconds) (pid $$), seeds=${SEEDS[*]} ====" | tee -a "${QUEUE_LOG}"

for seed in "${SEEDS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        name="hituav_exact_${shot}shot_seed${seed}"
        config_file="configs/hituav_exact_instance/vitl_shot${shot}_hituav_finetune_seed${seed}.yaml"
        proto_file="prototypes_init/hituav_exact_train_${shot}shot_seed${seed}.vitl14.bbox.p${shot}.sk.pkl"
        work_dir="${RESULT_DIR}/${name}"
        job_log="${work_dir}/result.log"
        done_marker="${work_dir}/.job_done"

        if [ ! -f "${config_file}" ]; then
            echo "[$(date -Iseconds)] FATAL: config not found: ${config_file}" | tee -a "${QUEUE_LOG}"
            exit 1
        fi
        if [ ! -f "${proto_file}" ]; then
            echo "[$(date -Iseconds)] FATAL: prototype not found: ${proto_file}" | tee -a "${QUEUE_LOG}"
            exit 1
        fi

        mkdir -p "${work_dir}"
        if [ -f "${done_marker}" ]; then
            echo "[$(date -Iseconds)] SKIP (already completed): ${name}" | tee -a "${QUEUE_LOG}"
            continue
        fi

        echo "[$(date -Iseconds)] START TRAIN+EVAL: ${name}" | tee -a "${QUEUE_LOG}"
        start=$(date +%s)
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 CUDA_VISIBLE_DEVICES=0 \
            python tools/train_net.py --num-gpus 1 \
            --config-file "${config_file}" \
            MODEL.WEIGHTS "${CKPT}" \
            DE.OFFLINE_RPN_CONFIG "${RPN_CFG}" \
            OUTPUT_DIR "${work_dir}/" \
            2>&1 | tee "${job_log}"
        exit_code=${PIPESTATUS[0]}
        end=$(date +%s)
        echo "[$(date -Iseconds)] END: ${name} exit_code=${exit_code} duration_s=$((end - start))" | tee -a "${QUEUE_LOG}"

        if [ "${exit_code}" -ne 0 ]; then
            echo "[$(date -Iseconds)] FATAL: job failed for ${name} (exit ${exit_code}), stopping queue." | tee -a "${QUEUE_LOG}"
            exit 1
        fi

        { echo "exit_code=${exit_code}"; echo "completed_at=$(date -Iseconds)"; } > "${done_marker}"
        echo "[$(date -Iseconds)] DONE: ${name}" | tee -a "${QUEUE_LOG}"
    done
done

echo "==== Queue finished successfully at $(date -Iseconds) ====" | tee -a "${QUEUE_LOG}"
