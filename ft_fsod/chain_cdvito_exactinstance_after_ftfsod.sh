#!/bin/bash
# Waits for the FT-FSOD exact-K-instance TXL-PBC queue (given seed) to
# finish, then (only on success) chains into CD-ViTO exact-instance:
# prototype build -> 3-job train+eval queue, same seed. TXL-PBC only, no
# HIT-UAV. Runs in its own tmux session so it survives this chat session
# ending, mirroring chain_cdvito_after_ftfsod.sh (the K-image counterpart).
#
# Usage: ./chain_cdvito_exactinstance_after_ftfsod.sh [SEED]   (default SEED=1)
set -uo pipefail

SEED="${1:-1}"
FT_DIR="/home/hjsjune/workspace/FT-FSOD"
CD_DIR="/home/hjsjune/workspace/CDFSOD-benchmark"
FT_LOG="${FT_DIR}/exp_exact_instance_results/queue_seed${SEED}.log"
CHAIN_LOG="${FT_DIR}/exp_exact_instance_results/chain_cdvito_seed${SEED}.log"

echo "[$(date -Iseconds)] Waiting for FT-FSOD exact-instance queue (${FT_LOG}) to finish..." | tee -a "${CHAIN_LOG}"
while ! grep -qE "Queue finished successfully|FATAL" "${FT_LOG}" 2>/dev/null; do
    sleep 30
done

if grep -q "FATAL" "${FT_LOG}"; then
    echo "[$(date -Iseconds)] FT-FSOD exact-instance queue FAILED - not starting CD-ViTO. See ${FT_LOG}" | tee -a "${CHAIN_LOG}"
    exit 1
fi

echo "[$(date -Iseconds)] FT-FSOD exact-instance queue (seed${SEED}) finished successfully." | tee -a "${CHAIN_LOG}"

echo "[$(date -Iseconds)] Checking GPU is idle before starting CD-ViTO..." | tee -a "${CHAIN_LOG}"
sleep 10  # let the last FT-FSOD process fully release GPU memory
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv | tee -a "${CHAIN_LOG}"

source /home/hjsjune/anaconda3/etc/profile.d/conda.sh
echo "[$(date -Iseconds)] Starting CD-ViTO exact-instance prototype build (seed${SEED})..." | tee -a "${CHAIN_LOG}"
cd "${CD_DIR}"
conda activate cdfsod
bash build_prototypes_txlpbc_exact_instance.sh "${SEED}" 2>&1 | tee -a "${CHAIN_LOG}"
proto_exit=${PIPESTATUS[0]}
if [ "${proto_exit}" -ne 0 ]; then
    echo "[$(date -Iseconds)] CD-ViTO exact-instance prototype build FAILED (exit ${proto_exit}) - not starting training queue." | tee -a "${CHAIN_LOG}"
    exit 1
fi

echo "[$(date -Iseconds)] Starting CD-ViTO exact-instance training queue (seed${SEED})..." | tee -a "${CHAIN_LOG}"
bash run_cdvito_txlpbc_exact_instance.sh "${SEED}" 2>&1 | tee -a "${CHAIN_LOG}"
cdvito_exit=${PIPESTATUS[0]}

echo "[$(date -Iseconds)] Chain complete (seed${SEED}). CD-ViTO queue exit=${cdvito_exit}" | tee -a "${CHAIN_LOG}"
