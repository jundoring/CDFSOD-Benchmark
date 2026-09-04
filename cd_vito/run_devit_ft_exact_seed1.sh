#!/bin/bash
# DE-ViT-FT on the exact-K-instance seed1 support sets, across all 3 datasets
# (TXL-PBC, SIXray-D, HIT-UAV) x {1,5,10}-shot - 9 jobs, matching the FSOD-VFM
# seed1 evaluation scope.
#
# Reuses the EXACT SAME finetune configs and prototype files as the CD-ViTO
# exact-instance runs (configs/txlpbc_exact_instance/, configs/
# sixray_exact_instance_seed1_clean/, configs/hituav/) - only difference from
# those CD-ViTO runs is passing --controller to train_net.py (sets
# DE.CONTROLLER=True, which disables CD-ViTO's extra trainable modules and
# reduces the model to the plain DE-ViT-FT baseline), matching this repo's
# existing run_devit_ft.sh recipe verbatim (same --controller flag, same
# TEST.EVAL_PERIOD 0 override - detectron2's EvalHook still runs exactly one
# evaluation at the end of training regardless of EVAL_PERIOD).
#
# Writes to output/vitl_devit_ft_exact_seed1/ - does not touch any existing
# CD-ViTO or prior DE-ViT output directories.
set -uo pipefail

RESULT_DIR="output/vitl_devit_ft_exact_seed1"
CKPT="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"

mkdir -p "${RESULT_DIR}"

# dataset:config_dir:config_prefix
JOBS_META=(
    "TXL-PBC:configs/txlpbc_exact_instance:vitl_shot{K}_txlpbc_finetune_seed1.yaml"
    "SIXray-D:configs/sixray_exact_instance_seed1_clean:vitl_shot{K}_sixray_finetune_seed1.yaml"
    "HIT-UAV:configs/hituav:vitl_shot{K}_hituav_finetune.yaml"
)
SHOTS=(1 5 10)

QUEUE_LOG="${RESULT_DIR}/queue_seed1.log"
echo "==== Queue started at $(date -Iseconds) (pid $$) ====" | tee -a "${QUEUE_LOG}"

for meta in "${JOBS_META[@]}"; do
    dataset="${meta%%:*}"
    rest="${meta#*:}"
    config_dir="${rest%%:*}"
    config_pattern="${rest#*:}"

    for shot in "${SHOTS[@]}"; do
        config_file="${config_dir}/${config_pattern//\{K\}/${shot}}"
        name="${dataset}_${shot}shot_seed1"
        work_dir="${RESULT_DIR}/${name}"
        job_log="${work_dir}/result.log"
        done_marker="${work_dir}/.job_done"

        if [ ! -f "${config_file}" ]; then
            echo "[$(date -Iseconds)] FATAL: config not found: ${config_file}" | tee -a "${QUEUE_LOG}"
            exit 1
        fi

        mkdir -p "${work_dir}"

        if [ -f "${done_marker}" ]; then
            echo "[$(date -Iseconds)] SKIP (already completed, see ${done_marker}): ${name}" | tee -a "${QUEUE_LOG}"
            continue
        fi

        echo "[$(date -Iseconds)] START DE-ViT-FT: ${name} config=${config_file}" | tee -a "${QUEUE_LOG}"
        start=$(date +%s)
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 CUDA_VISIBLE_DEVICES=0 \
            python tools/train_net.py --num-gpus 1 --controller \
            --config-file "${config_file}" \
            MODEL.WEIGHTS "${CKPT}" \
            DE.OFFLINE_RPN_CONFIG "${RPN_CFG}" \
            OUTPUT_DIR "${work_dir}/" \
            TEST.EVAL_PERIOD 0 \
            2>&1 | tee "${job_log}"
        exit_code=${PIPESTATUS[0]}
        end=$(date +%s)
        echo "[$(date -Iseconds)] END DE-ViT-FT: ${name} exit_code=${exit_code} duration_s=$((end - start))" | tee -a "${QUEUE_LOG}"

        if [ "${exit_code}" -ne 0 ]; then
            echo "[$(date -Iseconds)] FATAL: job failed for ${name} (exit ${exit_code}), stopping queue." | tee -a "${QUEUE_LOG}"
            exit 1
        fi

        {
            echo "exit_code=${exit_code}"
            echo "config_file=${config_file}"
            echo "completed_at=$(date -Iseconds)"
        } > "${done_marker}"

        echo "[$(date -Iseconds)] DONE: ${name}" | tee -a "${QUEUE_LOG}"
    done
done

echo "==== Queue finished successfully at $(date -Iseconds) ====" | tee -a "${QUEUE_LOG}"
