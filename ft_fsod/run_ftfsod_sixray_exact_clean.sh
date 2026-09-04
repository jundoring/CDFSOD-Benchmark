#!/bin/bash
# Fair-conditions SIXray-D exact-K-instance seed1 queue: 1/5/10-shot.
#
# Uses configs_cdfsod/final_configs_sixray_exact_seed1_clean/ (val_dataloader
# -> val.json, test_dataloader -> test.json, both disjoint from each other
# and from the train support set). Checkpoint selection (save_best=
# 'coco/bbox_mAP' in the config) happens purely against val.json throughout
# training; test.json is evaluated exactly once, after training, against the
# best-on-val checkpoint. Writes to exp_sixray_exact_seed1_clean/ - the
# existing exp_sixray_results/ tree and its configs are never touched.

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG_DIR="configs_cdfsod/final_configs_sixray_exact_seed1_clean"
RESULT_DIR="exp_sixray_exact_seed1_clean"
GPUID="0"
PORT="9997"

mkdir -p "${RESULT_DIR}"

JOBS=(1 5 10)

QUEUE_LOG="${RESULT_DIR}/queue_seed1.log"
echo "==== Queue started at $(date -Iseconds) (pid $$) ====" | tee -a "${QUEUE_LOG}"

for shot in "${JOBS[@]}"; do
    name="SIXray-D_${shot}shot_seed1"
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

    echo "[$(date -Iseconds)] START TRAIN (val-monitored): ${name}" | tee -a "${QUEUE_LOG}"
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
        echo "[$(date -Iseconds)] FATAL: no best-on-val checkpoint found for ${name} after training, stopping queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi

    echo "[$(date -Iseconds)] START TEST (single, held-out test.json): ${name} ckpt=${best_ckpt}" | tee -a "${QUEUE_LOG}"
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

echo "==== Queue finished successfully at $(date -Iseconds) ====" | tee -a "${QUEUE_LOG}"
