"""Convert the already-validated seed1 exact-K-instance COCO-format few-shot
JSONs (TXL-PBC, SIXray-D, HIT-UAV) into FSOD-VFM's official support-set
input format, as used by main.py / support_util.extract_support_features:

    {
      "class_name": [
        {"image": "<path relative to --data_dir>", "bbox": [x, y, w, h]},
        ...
      ],
      ...
    }

Each entry corresponds 1:1 to one annotation already selected by the
exact-K-instance sampler - no annotation not present in the source JSON is
ever added, and unselected annotations in the same source image are never
included (support_util.py only ever reads sample['bbox'], so there is no way
for it to pick up anything not explicitly listed here).

Source JSONs (read-only, never modified):
  TXL-PBC:  Dataset/TXL-PBC_Dataset/TXL-PBC/annotations_exact_instance/train_{K}shot_seed1.json
  SIXray-D: Dataset/SIXray-D/SIXray-D/fewshot/train_{K}shot_seed1.json
  HIT-UAV:  Dataset/HIT-UAV-Infrared-Thermal-Dataset/fewshot/train_{K}shot_seed1.json
            (DontCare already excluded upstream - only 4 official classes present)

Output: FSOD-VFM/data/{dataset}/exact_instance/{K}shot_seed1.json
        + a manifest.json alongside recording provenance (source path/SHA-256,
        class/count breakdown, output SHA-256, this script's path).

--data_dir at inference time must be each dataset's canonical root (passed
explicitly on the FSOD-VFM main.py command line), NOT this FSOD-VFM repo -
images are read directly from the canonical Dataset/ root and are never
copied or modified.
"""
import argparse
import hashlib
import json
import os
import subprocess

SHOTS = (1, 5, 10)
SEEDS = tuple(range(1, 11))

DATASETS = {
    "TXL-PBC": {
        "data_root": "/home/hjsjune/workspace/Dataset/TXL-PBC_Dataset/TXL-PBC",
        "support_src": lambda K, seed: f"annotations_exact_instance/train_{K}shot_seed{seed}.json",
        "image_prefix": "images/train/",
        "test_json": "annotations_std/test.json",
        "expected_classes": ["WBC", "RBC", "Platelets"],
    },
    "SIXray-D": {
        "data_root": "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D",
        "support_src": lambda K, seed: f"fewshot/train_{K}shot_seed{seed}.json",
        "image_prefix": "images/",
        "test_json": "test.json",
        "expected_classes": ["Gun", "Knife", "Wrench", "Pliers", "Scissors"],
    },
    "HIT-UAV": {
        "data_root": "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset",
        "support_src": lambda K, seed: f"fewshot/train_{K}shot_seed{seed}.json",
        "image_prefix": "normal_json/train/",
        "test_json": "normal_json/annotations_std/test.json",
        "expected_classes": ["Person", "Car", "Bicycle", "OtherVehicle"],
    },
}

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
THIS_SCRIPT = os.path.abspath(__file__)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUT_ROOT, text=True
        ).strip()
    except Exception:
        return None


def convert(dataset_name, K, seed, cfg):
    src_path = os.path.join(cfg["data_root"], cfg["support_src"](K, seed))
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    class_names = [cat_id_to_name[i] for i in sorted(cat_id_to_name)]
    if class_names != cfg["expected_classes"]:
        raise RuntimeError(
            f"{dataset_name}/{K}shot/seed{seed}: class order {class_names} != expected {cfg['expected_classes']}"
        )

    img_by_id = {im["id"]: im for im in data["images"]}

    support = {name: [] for name in class_names}
    per_class_count = {name: 0 for name in class_names}
    n_images_used = set()

    for ann in data["annotations"]:
        cls_name = cat_id_to_name[ann["category_id"]]
        file_name = img_by_id[ann["image_id"]]["file_name"]
        rel_path = cfg["image_prefix"] + file_name
        abs_img_path = os.path.join(cfg["data_root"], rel_path)
        if not os.path.isfile(abs_img_path):
            raise RuntimeError(f"{dataset_name}/{K}shot/seed{seed}: missing image file {abs_img_path}")
        support[cls_name].append({"image": rel_path, "bbox": list(ann["bbox"])})
        per_class_count[cls_name] += 1
        n_images_used.add(rel_path)

    for name in class_names:
        if per_class_count[name] != K:
            raise RuntimeError(
                f"{dataset_name}/{K}shot/seed{seed}: class '{name}' has {per_class_count[name]} "
                f"support instances, expected exactly {K}"
            )

    out_dir = os.path.join(OUT_ROOT, "data", dataset_name, "exact_instance")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{K}shot_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(support, f, ensure_ascii=False, indent=2)

    manifest_entry = {
        "dataset": dataset_name,
        "shot": K,
        "seed": seed,
        "source_json_abs_path": src_path,
        "source_json_sha256": sha256_of(src_path),
        "class_names": class_names,
        "class_instance_counts": per_class_count,
        "total_support_annotations": sum(per_class_count.values()),
        "total_support_images": len(n_images_used),
        "output_json_abs_path": out_path,
        "output_json_sha256": sha256_of(out_path),
        "data_root_for_inference": cfg["data_root"],
        "test_json_abs_path": os.path.join(cfg["data_root"], cfg["test_json"]),
        "test_json_sha256": sha256_of(os.path.join(cfg["data_root"], cfg["test_json"])),
        "generation_script": THIS_SCRIPT,
        "git_commit": git_commit(),
    }
    print(f"{dataset_name}/{K}shot_seed{seed}: images={len(n_images_used)} "
          f"annotations={sum(per_class_count.values())} per_class={per_class_count} -> {out_path}")
    return manifest_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    args = parser.parse_args()

    all_manifest = {"generated_by": THIS_SCRIPT, "git_commit": git_commit(), "entries": []}
    for dataset_name, cfg in DATASETS.items():
        for seed in args.seeds:
            for K in args.shots:
                entry = convert(dataset_name, K, seed, cfg)
                all_manifest["entries"].append(entry)

    seed_tag = "seed1" if args.seeds == [1] else f"seeds{min(args.seeds)}to{max(args.seeds)}"
    manifest_path = os.path.join(OUT_ROOT, "data", f"exact_instance_manifest_{seed_tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
