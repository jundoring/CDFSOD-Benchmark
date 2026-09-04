#!/bin/bash
# CD-ViTO exact-K-instance policy queue: TXL-PBC only, 1/5/10-shot, ViT-L
# backbone, one seed at a time (default seed1). Mirrors
# run_cdvito_txlpbc_hituav_seed1.sh but uses configs/txlpbc_exact_instance/
# and writes to a separate output/ tree so the K-image results are never
# touched. HIT-UAV is intentionally excluded.
#
# Each train_net.py call trains AND evaluates in one shot (TEST.EVAL_PERIOD
# == SOLVER.MAX_ITER, same protocol as the K-image queue).
#
# PREREQUISITE: prototypes_init/txlpbc_exact_train_{K}shot_seed{SEED}.vitl14.bbox.p{K}.sk.pkl
# must exist for all 3 jobs - run build_prototypes_txlpbc_exact_instance.sh <SEED> first.
#
# Usage: ./run_cdvito_txlpbc_exact_instance.sh [SEED]   (default SEED=1)
set -uo pipefail

SEED="${1:-1}"
RESULT_DIR="output/vitl_exact_instance_seed${SEED}"
CKPT="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"

mkdir -p "${RESULT_DIR}"

SHOTS=(1 5 10)

QUEUE_LOG="${RESULT_DIR}/queue_seed${SEED}.log"
echo "==== Queue started at $(date -Iseconds) (pid $$), seed=${SEED} ====" | tee -a "${QUEUE_LOG}"

for shot in "${SHOTS[@]}"; do
    name="txlpbc_exact_${shot}shot_seed${SEED}"
    config_file="configs/txlpbc_exact_instance/vitl_shot${shot}_txlpbc_finetune_seed${SEED}.yaml"
    proto_file="prototypes_init/txlpbc_exact_train_${shot}shot_seed${SEED}.vitl14.bbox.p${shot}.sk.pkl"
    work_dir="${RESULT_DIR}/${name}"
    job_log="${work_dir}/result.log"
    done_marker="${work_dir}/.job_done"

    if [ ! -f "${config_file}" ]; then
        echo "[$(date -Iseconds)] FATAL: config not found: ${config_file}" | tee -a "${QUEUE_LOG}"
        exit 1
    fi
    if [ ! -f "${proto_file}" ]; then
        echo "[$(date -Iseconds)] FATAL: prototype file not found: ${proto_file} (run build_prototypes_txlpbc_exact_instance.sh ${SEED} first)" | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    mkdir -p "${work_dir}"

    if [ -f "${done_marker}" ]; then
        echo "[$(date -Iseconds)] SKIP (already completed, see ${done_marker}): ${name}" | tee -a "${QUEUE_LOG}"
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
    echo "[$(date -Iseconds)] END TRAIN+EVAL: ${name} exit_code=${exit_code} duration_s=$((end - start))" | tee -a "${QUEUE_LOG}"

    if [ "${exit_code}" -ne 0 ]; then
        echo "[$(date -Iseconds)] FATAL: job failed for ${name} (exit ${exit_code}), stopping queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    {
        echo "exit_code=${exit_code}"
        echo "completed_at=$(date -Iseconds)"
    } > "${done_marker}"

    echo "[$(date -Iseconds)] DONE: ${name}" | tee -a "${QUEUE_LOG}"
done

echo "==== Queue finished successfully at $(date -Iseconds), seed=${SEED} ====" | tee -a "${QUEUE_LOG}"
