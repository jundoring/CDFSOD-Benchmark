"""Stage 7: aggregate the 9 per-job status.json files into the required
summary deliverables. mAP/AP50/AP75 are reported on a 0-100 scale for
cross-model comparability with the FT-FSOD/CD-ViTO results already reported
elsewhere in this project; the raw 0-1 COCOeval values are preserved
alongside in summary_seed1.json. Single seed (seed1) - no mean/std computed.
"""
import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(REPO_ROOT, "output")
SEED = 1
DATASETS = ["TXL-PBC", "SIXray-D", "HIT-UAV"]
SHOTS = [1, 5, 10]


def load_status(dataset, K):
    path = os.path.join(OUT_ROOT, dataset, "exact_instance", f"{K}shot", f"seed{SEED}", "status.json")
    with open(path) as f:
        return json.load(f)


def main():
    all_status = {ds: {K: load_status(ds, K) for K in SHOTS} for ds in DATASETS}

    # --- run_status_seed1.csv ---
    run_status_path = os.path.join(OUT_ROOT, f"run_status_seed{SEED}.csv")
    with open(run_status_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "shot", "seed", "status", "duration_s", "peak_vram_mib",
                    "n_test_images", "n_predictions", "AP_0_100", "AP50_0_100", "AP75_0_100"])
        for ds in DATASETS:
            for K in SHOTS:
                s = all_status[ds][K]
                w.writerow([ds, K, SEED, s["status"], f"{s['duration_s']:.1f}", f"{s['peak_vram_mib']:.1f}",
                            s["n_test_images"], s["n_predictions"],
                            f"{s['AP']*100:.2f}", f"{s['AP50']*100:.2f}", f"{s['AP75']*100:.2f}"])
    print(f"wrote {run_status_path}")

    # --- summary_overall_seed1.csv ---
    overall_path = os.path.join(OUT_ROOT, f"summary_overall_seed{SEED}.csv")
    with open(overall_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "Seed", "1-shot mAP", "5-shot mAP", "10-shot mAP",
                    "1-shot AP50", "5-shot AP50", "10-shot AP50", "1-shot AP75", "5-shot AP75", "10-shot AP75"])
        for ds in DATASETS:
            row = [ds, "FSOD-VFM", SEED]
            row += [f"{all_status[ds][K]['AP']*100:.2f}" for K in SHOTS]
            row += [f"{all_status[ds][K]['AP50']*100:.2f}" for K in SHOTS]
            row += [f"{all_status[ds][K]['AP75']*100:.2f}" for K in SHOTS]
            w.writerow(row)
    print(f"wrote {overall_path}")

    # --- summary_classwise_seed1.csv ---
    classwise_path = os.path.join(OUT_ROOT, f"summary_classwise_seed{SEED}.csv")
    with open(classwise_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Class", "1-shot AP", "5-shot AP", "10-shot AP"])
        for ds in DATASETS:
            class_names = list(all_status[ds][1]["class_wise_AP"].keys())
            for cls in class_names:
                row = [ds, cls]
                for K in SHOTS:
                    ap = all_status[ds][K]["class_wise_AP"][cls]
                    row.append(f"{ap*100:.2f}" if ap >= 0 else "N/A")
                w.writerow(row)
    print(f"wrote {classwise_path}")

    # --- summary_seed1.json (full detail, both scales) ---
    summary_json_path = os.path.join(OUT_ROOT, f"summary_seed{SEED}.json")
    full = {"seed": SEED, "note": "single-seed (seed1) results - no mean/std computed", "datasets": {}}
    for ds in DATASETS:
        full["datasets"][ds] = {}
        for K in SHOTS:
            s = all_status[ds][K]
            full["datasets"][ds][f"{K}shot"] = {
                "mAP_0_100": round(s["AP"] * 100, 4),
                "AP50_0_100": round(s["AP50"] * 100, 4),
                "AP75_0_100": round(s["AP75"] * 100, 4),
                "raw_coco_eval_0_1": {
                    "AP": s["AP"], "AP50": s["AP50"], "AP75": s["AP75"],
                    "APs": s["APs"], "APm": s["APm"], "APl": s["APl"],
                    "AR1": s["AR1"], "AR10": s["AR10"], "AR100": s["AR100"],
                },
                "class_wise_AP_0_100": {k: round(v * 100, 4) for k, v in s["class_wise_AP"].items()},
                "per_class_support_count": s["per_class_support_count"],
                "n_test_images": s["n_test_images"],
                "n_predictions": s["n_predictions"],
                "duration_s": s["duration_s"],
                "peak_vram_mib": s["peak_vram_mib"],
                "support_json_sha256": s["support_json_sha256"],
                "test_json_sha256": s["test_json_sha256"],
                "completed_at": s["completed_at"],
            }
    with open(summary_json_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"wrote {summary_json_path}")

    return all_status


if __name__ == "__main__":
    main()
