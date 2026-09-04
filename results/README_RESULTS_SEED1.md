# FSOD-VFM exact-K-instance seed1 results

Single-seed (**seed1 only**) results. No mean/std computed anywhere in this
report or its accompanying files - every table and value below applies to
seed1 alone.

Official FSOD-VFM pipeline (UPN -> SAM2.1 Hiera-Large -> DINOv2 ViT-L/14 ->
graph-diffusion reweighting), official CD-FSOD hyperparameters, unmodified:
`diffusion_steps=30, alp=0.3, lamb=0.5, min_threshold=0.01, points_per_side=32,
dinov2_image_size=630`. Training-free - no checkpoints, no train/val split,
support prototypes built once per (dataset, shot) then evaluated on the full,
unmodified, original test set for that dataset (identical test.json used by
the FT-FSOD/CD-ViTO experiments elsewhere in this project).

## Overall mAP (0-100 scale; raw 0-1 COCOeval values preserved in summary_seed1.json)

| Dataset | Model | Seed | 1-shot mAP | 5-shot mAP | 10-shot mAP |
|---|---|---|---|---|---|
| TXL-PBC | FSOD-VFM | 1 | 41.93 | 43.12 | 44.03 |
| SIXray-D | FSOD-VFM | 1 | 14.47 | 13.94 | 13.79 |
| HIT-UAV | FSOD-VFM | 1 | 7.49 | 11.70 | 14.12 |

## AP50 / AP75 (0-100 scale)

| Dataset | 1-shot AP50 | 5-shot AP50 | 10-shot AP50 | 1-shot AP75 | 5-shot AP75 | 10-shot AP75 |
|---|---|---|---|---|---|---|
| TXL-PBC | 62.47 | 66.02 | 66.24 | 49.11 | 49.42 | 50.98 |
| SIXray-D | 18.17 | 17.80 | 19.25 | 15.15 | 14.74 | 14.18 |
| HIT-UAV | 17.02 | 28.61 | 32.06 | 4.91 | 7.47 | 10.58 |

## Class-wise AP (0-100 scale)

| Dataset | Class | 1-shot AP | 5-shot AP | 10-shot AP |
|---|---|---|---|---|
| TXL-PBC | WBC | 70.30 | 73.33 | 73.68 |
| TXL-PBC | RBC | 52.66 | 50.02 | 53.58 |
| TXL-PBC | Platelets | 2.84 | 6.00 | 4.83 |
| SIXray-D | Gun | 55.26 | 54.84 | 52.82 |
| SIXray-D | Knife | 4.16 | 1.31 | 1.89 |
| SIXray-D | Wrench | 12.90 | 13.19 | 8.70 |
| SIXray-D | Pliers | 0.01 | 0.21 | 5.48 |
| SIXray-D | Scissors | 0.02 | 0.16 | 0.07 |
| HIT-UAV | Person | 0.00 | 15.71 | 15.42 |
| HIT-UAV | Car | 27.69 | 25.72 | 27.81 |
| HIT-UAV | Bicycle | 1.92 | 3.46 | 2.30 |
| HIT-UAV | OtherVehicle | 0.34 | 1.92 | 10.95 |

## Run status / cost

| Dataset | shot | status | duration | peak VRAM | test images | predictions |
|---|---|---|---|---|---|---|
| TXL-PBC | 1 | success | 82.0s | 4.80 GiB | 126 | 12,560 |
| TXL-PBC | 5 | success | 84.0s | 4.80 GiB | 126 | 12,560 |
| TXL-PBC | 10 | success | 86.7s | 4.80 GiB | 126 | 12,560 |
| SIXray-D | 1 | success | 618.7s | 5.24 GiB | 836 | 83,596 |
| SIXray-D | 5 | success | 630.8s | 5.24 GiB | 836 | 83,596 |
| SIXray-D | 10 | success | 635.0s | 5.24 GiB | 836 | 83,596 |
| HIT-UAV | 1 | success | 430.3s | 4.69 GiB | 579 | 57,896 |
| HIT-UAV | 5 | success | 432.3s | 4.69 GiB | 579 | 57,896 |
| HIT-UAV | 10 | success | 436.2s | 4.69 GiB | 579 | 57,896 |

9/9 jobs succeeded on the first attempt. No OOM anywhere (peak VRAM never
exceeded ~5.24 GiB of the 12 GiB card, all three models loaded simultaneously
consume ~3.0 GiB before any inference).

## Files

- `data/{dataset}/exact_instance/{K}shot_seed1.json` - official-format converted support sets
- `data/exact_instance_manifest_seed1.json` - conversion manifest (source/output SHA-256, per-class counts)
- `output/{dataset}/exact_instance/{K}shot/seed1/run_args.json` - full CLI-equivalent arguments used
- `output/{dataset}/exact_instance/{K}shot/seed1/predictions.json` - raw prediction JSON
- `output/{dataset}/exact_instance/{K}shot/seed1/status.json` - per-job metrics/status (source of truth for this report)
- `output/run_status_seed1.csv`, `summary_overall_seed1.csv`, `summary_classwise_seed1.csv`, `summary_seed1.json` - aggregated results
- `output/smoke_test_TXL-PBC_1shot_seed1/` - smoke-test support crops and prediction visualizations
