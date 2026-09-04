#!/bin/bash
# DE-ViT-FT on exact-K-instance support sets, seeds 2..10, across all 3
# datasets (TXL-PBC, SIXray-D, HIT-UAV) x {1,5,10}-shot - 81 jobs by default.
# seed1 is excluded by default (already run via run_devit_ft_exact_seed1.sh).
#
# Reuses the exact same finetune configs as the CD-ViTO multiseed runs
# (configs/txlpbc_exact_instance/, configs/sixray_exact_instance_seed1_clean/,
# configs/hituav_exact_instance/) with --controller (matches
# run_devit_ft_exact_seed1.sh's recipe). Writes to
# output/vitl_devit_ft_exact_multiseed/ - does not touch the seed1 output.
#
# Usage: ./run_devit_ft_exact_multiseed.sh [SEED...]  (default: 2..10)
set -uo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(2 3 4 5 6 7 8 9 10)
fi

RESULT_DIR="output/vitl_devit_ft_exact_multiseed"
CKPT="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"
SHOTS=(1 5 10)

mkdir -p "${RESULT_DIR}"
QUEUE_LOG="${RESULT_DIR}/queue.log"
echo "==== Queue started at $(date -Iseconds) (pid $$), seeds=${SEEDS[*]} ====" | tee -a "${QUEUE_LOG}"

# dataset:config_dir:config_pattern
JOBS_META=(
    "TXL-PBC:configs/txlpbc_exact_instance:vitl_shot{K}_txlpbc_finetune_seed{S}.yaml"
    "SIXray-D:configs/sixray_exact_instance_seed1_clean:vitl_shot{K}_sixray_finetune_seed{S}.yaml"
    "HIT-UAV:configs/hituav_exact_instance:vitl_shot{K}_hituav_finetune_seed{S}.yaml"
)

for seed in "${SEEDS[@]}"; do
    for meta in "${JOBS_META[@]}"; do
        dataset="${meta%%:*}"
        rest="${meta#*:}"
        config_dir="${rest%%:*}"
        config_pattern="${rest#*:}"

        for shot in "${SHOTS[@]}"; do
            pattern="${config_pattern//\{K\}/${shot}}"
            config_file="${config_dir}/${pattern//\{S\}/${seed}}"
            name="${dataset}_${shot}shot_seed${seed}"
            work_dir="${RESULT_DIR}/${name}"
            job_log="${work_dir}/result.log"
            done_marker="${work_dir}/.job_done"

            if [ ! -f "${config_file}" ]; then
                echo "[$(date -Iseconds)] FATAL: config not found: ${config_file}" | tee -a "${QUEUE_LOG}"
                exit 1
            fi

            mkdir -p "${work_dir}"
            if [ -f "${done_marker}" ]; then
                echo "[$(date -Iseconds)] SKIP (already completed): ${name}" | tee -a "${QUEUE_LOG}"
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
            echo "[$(date -Iseconds)] END: ${name} exit_code=${exit_code} duration_s=$((end - start))" | tee -a "${QUEUE_LOG}"

            if [ "${exit_code}" -ne 0 ]; then
                echo "[$(date -Iseconds)] FATAL: job failed for ${name} (exit ${exit_code}), stopping queue." | tee -a "${QUEUE_LOG}"
                exit 1
            fi

            { echo "exit_code=${exit_code}"; echo "config_file=${config_file}"; echo "completed_at=$(date -Iseconds)"; } > "${done_marker}"
            echo "[$(date -Iseconds)] DONE: ${name}" | tee -a "${QUEUE_LOG}"
        done
    done
done

echo "==== Queue finished successfully at $(date -Iseconds) ====" | tee -a "${QUEUE_LOG}"
