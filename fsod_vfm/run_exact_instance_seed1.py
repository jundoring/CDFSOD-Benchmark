"""Stage 6: run all 9 (dataset x shot) exact-K-instance seed1 evaluations
using the official FSOD-VFM pipeline (UPN -> SAM2 -> DINOv2 -> graph
diffusion, via the unmodified support_util.py / query_util.py / metric.py
functions), official CD-FSOD hyperparameters (diffusion_steps=30, alp=0.3,
lamb=0.5, min_threshold=0.01, points_per_side=32, dinov2_vitl14 @ 630px,
sam2.1 hiera-large). Training-free - no checkpoints, no train/val split.

Models are loaded ONCE and reused across all 9 jobs (a performance choice;
each job still runs the identical official forward-pass functions per image,
so results are unaffected - this only saves ~4s of repeated model loading
per job that main.py would otherwise pay for each separate CLI invocation).

Resumable: each job's output dir gets a status.json with status=success only
after a fully completed run; already-successful jobs are skipped on rerun.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch
import pycocotools.coco

import model.dinov2
import model.sam2
import support_util
import query_util
import metric
from chatrex.upn import UPNWrapper

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "TXL-PBC": {
        "data_root": "/home/hjsjune/workspace/Dataset/TXL-PBC_Dataset/TXL-PBC",
        "test_json": "annotations_std/test.json",
        "test_img_dir": "images/test",
        "target_categories": ["WBC", "RBC", "Platelets"],
    },
    "SIXray-D": {
        "data_root": "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D",
        "test_json": "test.json",
        "test_img_dir": "images",
        "target_categories": ["Gun", "Knife", "Wrench", "Pliers", "Scissors"],
    },
    "HIT-UAV": {
        "data_root": "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset",
        "test_json": "normal_json/annotations_std/test.json",
        "test_img_dir": "normal_json/test",
        "target_categories": ["Person", "Car", "Bicycle", "OtherVehicle"],
    },
}
SHOTS = (1, 5, 10)

HPARAMS = dict(
    feat_extractor_name="DINOV2",
    model_version="dinov2_vitl14",
    dinov2_image_size=630,
    sam2_model_type="large",
    points_per_side=32,
    min_threshold=0.01,
    diffusion_steps=30,
    alp=0.3,
    lamb=0.5,
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def class_wise_ap(coco_gt, target_categories, precision):
    """Replicates pycocotools COCOeval.summarize()'s per-category AP@[0.5:0.95]
    (area='all' idx 0, maxDets=100 -> idx 2), using the same catIds order the
    official metric.run_coco_eval used internally (deterministic given the
    same gt file and target_categories list)."""
    cat_ids = coco_gt.getCatIds(catNms=target_categories)
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    precision = np.array(precision)  # [T, R, K, A, M]
    out = {}
    for k, cid in enumerate(cat_ids):
        s = precision[:, :, k, 0, 2]
        s = s[s > -1]
        out[id_to_name[cid]] = float(np.mean(s)) if s.size else -1.0
    return out


def run_one(dataset_name, K, seed, upn, feat_extractor, image_transform,
            sam2_model, sam2_predictor, sam2_mask_generator):
    cfg = DATASETS[dataset_name]
    support_json = os.path.join(REPO_ROOT, "data", dataset_name, "exact_instance", f"{K}shot_seed{seed}.json")
    test_json = os.path.join(cfg["data_root"], cfg["test_json"])
    test_img_dir = os.path.join(cfg["data_root"], cfg["test_img_dir"])

    out_dir = os.path.join(REPO_ROOT, "output", dataset_name, "exact_instance", f"{K}shot", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "status.json")

    if os.path.isfile(status_path):
        with open(status_path) as f:
            prior = json.load(f)
        if prior.get("status") == "success":
            print(f"[SKIP] {dataset_name}/{K}shot/seed{seed} already succeeded ({status_path})")
            return prior

    args_record = dict(
        dataset=dataset_name, shot=K, seed=seed,
        support_json=support_json, support_json_sha256=sha256_of(support_json),
        test_json=test_json, test_json_sha256=sha256_of(test_json),
        test_img_dir=test_img_dir, data_dir=cfg["data_root"],
        target_categories=cfg["target_categories"], filter_by_categories=True,
        device="cuda", pretrained=f"./checkpoints/dinov2_vitl14_pretrain.pth",
        repo_or_dir="./dinov2", dinov2_checkpoint_dir="./checkpoints",
        git_commit=git_commit(), **HPARAMS,
    )
    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        json.dump(args_record, f, indent=2)

    print(f"\n{'='*70}\n[RUN] {dataset_name}/{K}shot/seed{seed}\n{'='*70}")
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    try:
        with open(support_json) as f:
            support_data = json.load(f)
        per_class_count = {c: len(v) for c, v in support_data.items()}
        assert all(v == K for v in per_class_count.values()), \
            f"expected {K} per class, got {per_class_count}"

        memory_bank = support_util.extract_support_features(
            support_data, sam2_predictor, "DINOV2", feat_extractor, image_transform,
            cfg["data_root"], "cuda",
        )
        proto_feat, proto_cls = support_util.compute_prototype_weights(memory_bank, "cuda")
        assert not torch.isnan(proto_feat).any().item() and not torch.isinf(proto_feat).any().item()

        image_paths, coco_style_loader = query_util.load_voc2007_coco_json(test_json, test_img_dir)

        results = metric.generate_coco_style_predictions_upn(
            coco_style_loader, test_img_dir, sam2_predictor, "DINOV2", feat_extractor,
            image_transform, proto_feat, proto_cls, upn,
            HPARAMS["diffusion_steps"], HPARAMS["alp"], HPARAMS["lamb"], "cuda",
            HPARAMS["min_threshold"],
        )
        torch.cuda.synchronize()
        n_images = len(image_paths)
        images_with_pred = len({r["image_id"] for r in results})

        pred_json_path = os.path.join(out_dir, "predictions.json")
        eval_results = metric.run_coco_eval(
            test_json, results, pred_json=pred_json_path,
            target_categories=cfg["target_categories"], filter_by_categories=True,
        )

        coco_gt = pycocotools.coco.COCO(test_json)
        cw_ap = class_wise_ap(coco_gt, cfg["target_categories"], eval_results["bbox"]["precision"])

        duration_s = time.time() - t_start
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        stats = eval_results["bbox"]["stats"]

        status = dict(
            dataset=dataset_name, shot=K, seed=seed, status="success",
            duration_s=duration_s, peak_vram_mib=peak_vram,
            n_test_images=n_images, images_with_predictions=images_with_pred,
            n_predictions=len(results),
            AP=stats[0], AP50=stats[1], AP75=stats[2],
            APs=stats[3], APm=stats[4], APl=stats[5],
            AR1=stats[6], AR10=stats[7], AR100=stats[8],
            class_wise_AP=cw_ap,
            per_class_support_count=per_class_count,
            support_json_sha256=args_record["support_json_sha256"],
            test_json_sha256=args_record["test_json_sha256"],
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)
        print(f"[SUCCESS] {dataset_name}/{K}shot/seed{seed}: AP={stats[0]:.4f} AP50={stats[1]:.4f} "
              f"AP75={stats[2]:.4f} duration={duration_s:.1f}s peak_vram={peak_vram:.0f}MiB")
        return status

    except Exception as e:
        duration_s = time.time() - t_start
        err = dict(dataset=dataset_name, shot=K, seed=seed, status="failure",
                   duration_s=duration_s, error=str(e), traceback=traceback.format_exc())
        with open(status_path, "w") as f:
            json.dump(err, f, indent=2)
        print(f"[FAILURE] {dataset_name}/{K}shot/seed{seed}: {e}")
        return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--shots", nargs="+", type=int, default=list(SHOTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    print("Loading UPN...")
    upn = UPNWrapper("./checkpoints/upn_large.pth")
    print("Loading DINOv2...")
    feat_extractor, image_transform = model.dinov2.load_dinov2_model(
        "cuda", HPARAMS["model_version"],
        image_size=(HPARAMS["dinov2_image_size"], HPARAMS["dinov2_image_size"]),
        repo_or_dir="./dinov2", pretrained="./checkpoints/dinov2_vitl14_pretrain.pth",
    )
    print("Loading SAM2...")
    sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
        model_type=HPARAMS["sam2_model_type"], device="cuda", points_per_side=HPARAMS["points_per_side"],
    )

    all_status = []
    for seed in args.seeds:
        for dataset_name in args.datasets:
            for K in args.shots:
                status = run_one(dataset_name, K, seed, upn, feat_extractor, image_transform,
                                  sam2_model, sam2_predictor, sam2_mask_generator)
                all_status.append(status)

    seed_tag = "seed1" if args.seeds == [1] else f"seeds{min(args.seeds)}to{max(args.seeds)}"
    out_summary = os.path.join(REPO_ROOT, "output", f"run_status_{seed_tag}_raw.json")
    with open(out_summary, "w") as f:
        json.dump(all_status, f, indent=2)
    print(f"\nAll jobs done. Raw status summary -> {out_summary}")


if __name__ == "__main__":
    main()
