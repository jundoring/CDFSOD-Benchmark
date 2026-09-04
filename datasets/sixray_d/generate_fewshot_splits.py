"""SIXray-D exact-K-instance few-shot split generator.

NOTE ON ORIGIN: the per-class sampling algorithm below is a direct
reimplementation of `datasets/kshot_split.py` from the original
CDFSOD-benchmark repository (https://github.com/lovelyqian/CDFSOD-benchmark),
which is the algorithm this project's SIXray-D exact-instance splits were
built with. It is NOT an original algorithm invented by this project - it is
reproduced here so a licensed copy of SIXray-D can be turned into the same
K-shot splits without pulling in the full CDFSOD-benchmark checkout. Unlike
TXL-PBC/HIT-UAV in this project (data_generation/build_txlpbc_exact_kinstance.py,
data_generation/build_hituav_exact_kinstance_seed2to10.py), classes are
sampled independently per class - an image CAN be reused across classes, only
the K instances per class are guaranteed exact.

This script never ships or embeds any SIXray-D annotation data. It only reads
a `train.json` (COCO format) that YOU provide from your own licensed copy of
SIXray-D, and writes `train_{K}shot_seed{N}.json` files next to it.
"""
import argparse
import json
import os
import random
from collections import Counter

SHOTS = (1, 5, 10)
SEEDS = range(1, 11)


def sample_k_shot(data, k_shot, seed):
    """Reimplementation of CDFSOD-benchmark's filter_k_shot_json: independent
    per-category sampling, image reuse across categories allowed."""
    rng = random.Random(seed)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    image_id_to_image = {im["id"]: im for im in images}

    category_to_image_annotations = {}
    for ann in annotations:
        cid, image_id = ann["category_id"], ann["image_id"]
        category_to_image_annotations.setdefault(cid, {})
        category_to_image_annotations[cid].setdefault(image_id, ann)

    selected_annotations = []
    selected_image_ids = set()
    for cid in sorted(category_to_image_annotations):
        image_ids = sorted(category_to_image_annotations[cid])
        chosen = image_ids if len(image_ids) <= k_shot else rng.sample(image_ids, k_shot)
        selected_image_ids.update(chosen)
        selected_annotations.extend(category_to_image_annotations[cid][iid] for iid in chosen)

    selected_images = [image_id_to_image[iid] for iid in sorted(selected_image_ids)]
    return {"images": selected_images, "annotations": selected_annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, help="path to your local SIXray-D train.json (COCO format)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.train_json, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in SEEDS:
        for k in SHOTS:
            split = sample_k_shot(data, k, seed)
            out_path = os.path.join(args.out_dir, f"train_{k}shot_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(split, f, ensure_ascii=False)
            counts = Counter(a["category_id"] for a in split["annotations"])
            print(f"seed {seed}, {k}-shot: {len(split['images'])} images, "
                  f"{len(split['annotations'])} annotations, per-class={dict(counts)} -> {out_path}")


if __name__ == "__main__":
    main()
