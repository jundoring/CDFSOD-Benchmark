#!/bin/bash
# Fair-conditions CD-ViTO SIXray-D exact-K-instance seed1 queue: 1/5/10-shot,
# ViT-L backbone. Uses configs/sixray_exact_instance_seed1_clean/ (canonical
# Dataset/SIXray-D root, sixray_exact_train_{K}shot_seed1 / sixray_exact_test).
# Each train_net.py call trains AND evaluates in one shot (TEST.EVAL_PERIOD ==
# SOLVER.MAX_ITER, same protocol used throughout this codebase - detectron2
# side has no separate best-val-checkpoint-selection step). Writes to
# output/vitl_sixray_exact_seed1_clean/ - existing output/vitl/sixray_*
# results are never touched.
#
# PREREQUISITE: prototypes_init/sixray_exact_train_{K}shot_seed1.vitl14.bbox.p{K}.sk.pkl
# must exist for all 3 jobs - run build_prototypes_sixray_exact_clean.sh first.
set -uo pipefail

RESULT_DIR="output/vitl_sixray_exact_seed1_clean"
CKPT="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"

mkdir -p "${RESULT_DIR}"

SHOTS=(1 5 10)

QUEUE_LOG="${RESULT_DIR}/queue_seed1.log"
echo "==== Queue started at $(date -Iseconds) (pid $$) ====" | tee -a "${QUEUE_LOG}"

for shot in "${SHOTS[@]}"; do
    name="sixray_exact_${shot}shot_seed1"
    config_file="configs/sixray_exact_instance_seed1_clean/vitl_shot${shot}_sixray_finetune_seed1.yaml"
    proto_file="prototypes_init/sixray_exact_train_${shot}shot_seed1.vitl14.bbox.p${shot}.sk.pkl"
    work_dir="${RESULT_DIR}/${name}"
    job_log="${work_dir}/result.log"
    done_marker="${work_dir}/.job_done"

    if [ ! -f "${config_file}" ]; then
        echo "[$(date -Iseconds)] FATAL: config not found: ${config_file}" | tee -a "${QUEUE_LOG}"
        exit 1
    fi
    if [ ! -f "${proto_file}" ]; then
        echo "[$(date -Iseconds)] FATAL: prototype file not found: ${proto_file} (run build_prototypes_sixray_exact_clean.sh first)" | tee -a "${QUEUE_LOG}"
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

echo "==== Queue finished successfully at $(date -Iseconds) ====" | tee -a "${QUEUE_LOG}"
