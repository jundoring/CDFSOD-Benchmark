#!/bin/bash
# Waits for the FT-FSOD SIXray-D exact-instance clean queue to finish, then
# (only on success) builds CD-ViTO prototypes and runs its 3-job queue.
set -uo pipefail

FT_DIR="/home/hjsjune/workspace/FT-FSOD"
CD_DIR="/home/hjsjune/workspace/CDFSOD-benchmark"
FT_LOG="${FT_DIR}/exp_sixray_exact_seed1_clean/queue_seed1.log"
CHAIN_LOG="${FT_DIR}/exp_sixray_exact_seed1_clean/chain_cdvito.log"

echo "[$(date -Iseconds)] Waiting for FT-FSOD SIXray-D exact-clean queue (${FT_LOG})..." | tee -a "${CHAIN_LOG}"
while ! grep -qE "Queue finished successfully|FATAL" "${FT_LOG}" 2>/dev/null; do
    sleep 30
done

if grep -q "FATAL" "${FT_LOG}"; then
    echo "[$(date -Iseconds)] FT-FSOD queue FAILED - not starting CD-ViTO. See ${FT_LOG}" | tee -a "${CHAIN_LOG}"
    exit 1
fi

echo "[$(date -Iseconds)] FT-FSOD queue finished successfully." | tee -a "${CHAIN_LOG}"
sleep 10
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv | tee -a "${CHAIN_LOG}"

source /home/hjsjune/anaconda3/etc/profile.d/conda.sh
echo "[$(date -Iseconds)] Starting CD-ViTO SIXray-D exact-clean prototype build..." | tee -a "${CHAIN_LOG}"
cd "${CD_DIR}"
conda activate cdfsod
bash build_prototypes_sixray_exact_clean.sh 2>&1 | tee -a "${CHAIN_LOG}"
proto_exit=${PIPESTATUS[0]}
if [ "${proto_exit}" -ne 0 ]; then
    echo "[$(date -Iseconds)] CD-ViTO prototype build FAILED (exit ${proto_exit})." | tee -a "${CHAIN_LOG}"
    exit 1
fi

echo "[$(date -Iseconds)] Starting CD-ViTO SIXray-D exact-clean training queue..." | tee -a "${CHAIN_LOG}"
bash run_cdvito_sixray_exact_clean.sh 2>&1 | tee -a "${CHAIN_LOG}"
cdvito_exit=${PIPESTATUS[0]}

echo "[$(date -Iseconds)] Chain complete. CD-ViTO queue exit=${cdvito_exit}" | tee -a "${CHAIN_LOG}"
