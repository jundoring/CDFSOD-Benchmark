#!/bin/bash
# Master orchestrator: extends all 4 models (FSOD-VFM, DE-ViT-FT, CD-ViTO,
# FT-FSOD) to exact-K-instance seed2..10 across TXL-PBC/SIXray-D/HIT-UAV,
# skipping anything already completed (TXL-PBC FT-FSOD/CD-ViTO seed1-10, and
# every model's existing seed1 results). Runs in cost-ascending order so the
# cheapest, fastest results land first:
#   1. FSOD-VFM        (~8.5h)  - training-free, all 3 datasets
#   2. Build prototypes (~1-2h) - SIXray-D + HIT-UAV seed2-10 (needed by BOTH
#                                 DE-ViT-FT and CD-ViTO below - must happen
#                                 before either, this is a fix after DE-ViT-FT
#                                 failed once trying to load a prototype file
#                                 that CD-ViTO's stage would only have built
#                                 afterward under the old ordering)
#   3. DE-ViT-FT        (~21h)  - all 3 datasets
#   4. CD-ViTO          (~20h)  - SIXray-D + HIT-UAV only (TXL-PBC done),
#                                 prototypes already built in step 2
#   5. FT-FSOD          (~8.4d) - HIT-UAV then SIXray-D (TXL-PBC done)
# Total estimate ~10.5 days, single GPU, sequential. Stops immediately on
# any stage's failure - does not proceed to the next model. All sub-scripts
# have their own .job_done/status.json resume logic, so re-running this
# script after a fix or interruption picks up where it left off.
set -uo pipefail

FT_DIR="/home/hjsjune/workspace/FT-FSOD"
FSODVFM_DIR="/home/hjsjune/workspace/FSOD-VFM"
CD_DIR="/home/hjsjune/workspace/CDFSOD-benchmark"
MASTER_LOG="${FT_DIR}/exp_exact_multiseed_master.log"

source /home/hjsjune/anaconda3/etc/profile.d/conda.sh

log() { echo "[$(date -Iseconds)] $*" | tee -a "${MASTER_LOG}"; }

log "==== all_exact_multiseed master started (pid $$) ===="

# ---- 1. FSOD-VFM ----
log "==== STAGE 1/5: FSOD-VFM seed2-10 (all 3 datasets) starting ===="
cd "${FSODVFM_DIR}"
conda run -n FSODVFM python run_exact_instance_seed1.py --seeds 2 3 4 5 6 7 8 9 10 2>&1 | tee -a "${MASTER_LOG}"
fsodvfm_exit=${PIPESTATUS[0]}
if [ "${fsodvfm_exit}" -ne 0 ]; then
    log "FATAL: FSOD-VFM stage failed (exit ${fsodvfm_exit}). Stopping."
    exit 1
fi
log "STAGE 1/5 done: FSOD-VFM."

# ---- 2. Prototypes (SIXray-D + HIT-UAV seed2-10) - shared by DE-ViT-FT and CD-ViTO ----
log "==== STAGE 2/5: build prototypes (SIXray-D + HIT-UAV seed2-10) starting ===="
sleep 10
cd "${CD_DIR}"
conda activate cdfsod
bash build_prototypes_sixray_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
proto1_exit=${PIPESTATUS[0]}
if [ "${proto1_exit}" -ne 0 ]; then
    log "FATAL: SIXray-D prototype build failed (exit ${proto1_exit}). Stopping."
    exit 1
fi
bash build_prototypes_hituav_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
proto2_exit=${PIPESTATUS[0]}
if [ "${proto2_exit}" -ne 0 ]; then
    log "FATAL: HIT-UAV prototype build failed (exit ${proto2_exit}). Stopping."
    exit 1
fi
log "STAGE 2/5 done: prototypes."

# ---- 3. DE-ViT-FT ----
log "==== STAGE 3/5: DE-ViT-FT seed2-10 (all 3 datasets) starting ===="
sleep 10
cd "${CD_DIR}"
conda activate cdfsod
bash run_devit_ft_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
devit_exit=${PIPESTATUS[0]}
if [ "${devit_exit}" -ne 0 ]; then
    log "FATAL: DE-ViT-FT stage failed (exit ${devit_exit}). Stopping."
    exit 1
fi
log "STAGE 3/5 done: DE-ViT-FT."

# ---- 4. CD-ViTO (SIXray-D + HIT-UAV) ----
log "==== STAGE 4/5: CD-ViTO seed2-10 (SIXray-D + HIT-UAV) starting ===="
sleep 10
cd "${CD_DIR}"
conda activate cdfsod
bash run_cdvito_sixray_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
cdvito_sixray_exit=${PIPESTATUS[0]}
if [ "${cdvito_sixray_exit}" -ne 0 ]; then
    log "FATAL: CD-ViTO SIXray-D stage failed (exit ${cdvito_sixray_exit}). Stopping."
    exit 1
fi
bash run_cdvito_hituav_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
cdvito_hituav_exit=${PIPESTATUS[0]}
if [ "${cdvito_hituav_exit}" -ne 0 ]; then
    log "FATAL: CD-ViTO HIT-UAV stage failed (exit ${cdvito_hituav_exit}). Stopping."
    exit 1
fi
log "STAGE 4/5 done: CD-ViTO."

# ---- 5. FT-FSOD (HIT-UAV then SIXray-D) ----
log "==== STAGE 5/5: FT-FSOD seed2-10 (HIT-UAV then SIXray-D) starting ===="
sleep 10
cd "${FT_DIR}"
conda activate ft-fsod
./run_ftfsod_hituav_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
ft_hituav_exit=${PIPESTATUS[0]}
if [ "${ft_hituav_exit}" -ne 0 ]; then
    log "FATAL: FT-FSOD HIT-UAV stage failed (exit ${ft_hituav_exit}). Stopping."
    exit 1
fi
./run_ftfsod_sixray_exact_multiseed.sh 2>&1 | tee -a "${MASTER_LOG}"
ft_sixray_exit=${PIPESTATUS[0]}
if [ "${ft_sixray_exit}" -ne 0 ]; then
    log "FATAL: FT-FSOD SIXray-D stage failed (exit ${ft_sixray_exit}). Stopping."
    exit 1
fi
log "STAGE 5/5 done: FT-FSOD."

log "==== all_exact_multiseed master FINISHED SUCCESSFULLY (all 5 stages) ===="
