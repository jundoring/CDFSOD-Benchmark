#!/bin/bash
# FT-FSOD exact-K-instance policy queue: TXL-PBC only, 1/5/10-shot, one seed
# at a time (default seed1). HIT-UAV is intentionally excluded - the
# exact-instance policy change is TXL-PBC-only.
#
# Counterpart of run_ftfsod_preliminary_traineval_seed1.sh (K-image policy),
# but points at configs_cdfsod/final_configs_exact_instance/ and writes to a
# separate exp_exact_instance_results/ tree so neither the K-image configs
# nor their results are ever touched.
#
# Usage: ./run_ftfsod_exactinstance_traineval.sh [SEED]   (default SEED=1)

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SEED="${1:-1}"
CONFIG_DIR="configs_cdfsod/final_configs_exact_instance"
RESULT_DIR="exp_exact_instance_results"
GPUID="0"
PORT="9996"

mkdir -p "${RESULT_DIR}"

JOBS=(1 5 10)

QUEUE_LOG="${RESULT_DIR}/queue_seed${SEED}.log"
echo "==== Queue started at $(date -Iseconds) (pid $$), seed=${SEED} ====" | tee -a "${QUEUE_LOG}"

for shot in "${JOBS[@]}"; do
    name="TXL-PBC_${shot}shot_seed${SEED}"
    config_file="${CONFIG_DIR}/grounding_dino_swin-b_finetune_${name}.py"
    work_dir="${RESULT_DIR}/swinB_${name}"
    job_log="${work_dir}/job.log"
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

    echo "[$(date -Iseconds)] START TRAIN: ${name}" | tee -a "${QUEUE_LOG}"
    train_start=$(date +%s)
    ./tools/dist_train.sh "${config_file}" 1 "${PORT}" "${GPUID}" --work-dir "${work_dir}" 2>&1 | tee -a "${job_log}"
    train_exit=${PIPESTATUS[0]}
    train_end=$(date +%s)
    echo "[$(date -Iseconds)] END TRAIN: ${name} exit_code=${train_exit} duration_s=$((train_end - train_start))" | tee -a "${QUEUE_LOG}"

    if [ "${train_exit}" -ne 0 ]; then
        echo "[$(date -Iseconds)] FATAL: training failed for ${name} (exit ${train_exit}), stopping queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    best_ckpt=$(find "${work_dir}" -maxdepth 1 -name "best_coco_bbox_mAP_iter_*.pth" | sort -V | tail -n1)
    if [ -z "${best_ckpt}" ]; then
        echo "[$(date -Iseconds)] FATAL: no best checkpoint found for ${name} after training, stopping queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    echo "[$(date -Iseconds)] START TEST: ${name} ckpt=${best_ckpt}" | tee -a "${QUEUE_LOG}"
    test_start=$(date +%s)
    ./tools/dist_test.sh "${config_file}" "${best_ckpt}" 1 "${PORT}" "${GPUID}" --work-dir "${work_dir}" --out "${work_dir}/${name}_test.pkl" 2>&1 | tee -a "${job_log}"
    test_exit=${PIPESTATUS[0]}
    test_end=$(date +%s)
    echo "[$(date -Iseconds)] END TEST: ${name} exit_code=${test_exit} duration_s=$((test_end - test_start))" | tee -a "${QUEUE_LOG}"

    if [ "${test_exit}" -ne 0 ]; then
        echo "[$(date -Iseconds)] FATAL: test failed for ${name} (exit ${test_exit}), stopping queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    {
        echo "best_ckpt=${best_ckpt}"
        echo "train_exit=${train_exit}"
        echo "test_exit=${test_exit}"
        echo "completed_at=$(date -Iseconds)"
    } > "${done_marker}"

    echo "[$(date -Iseconds)] DONE: ${name}" | tee -a "${QUEUE_LOG}"
done

echo "==== Queue finished successfully at $(date -Iseconds), seed=${SEED} ====" | tee -a "${QUEUE_LOG}"
