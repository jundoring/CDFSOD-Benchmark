"""TXL-PBC few-shot generation under the official CD-FSOD exact-K-instance policy.

This is a NEW, separate policy from the existing "K-image" support_sets/fewshot/
annotations_std/train_{K}shot_seed1.json files (which pick K whole images per
seed from a shared "all-3-classes-present" pool and keep every annotation in
each picked image). Those files are read-only inputs to this script and are
never modified or deleted.

Policy (matches CDFSOD-benchmark/datasets/kshot_split.py's per-category exact-K
selection, made strict about cross-category image disjointness):
  - Sample only from the original TXL-PBC `train` split (labels/train, images/train).
  - Fixed class processing/output order: WBC(0), RBC(1), Platelets(2).
  - For K in {1, 5, 10}: exactly K annotations per class, i.e. 3*K annotations
    and 3*K images total (one annotation per image, no image reused across
    classes or within a class).
  - Per class: build the candidate pool of train images containing >=1
    annotation of that class (excluding images already claimed by an
    earlier-processed class in the same seed), sort image stems for a
    deterministic starting order, then seed a `random.Random(seed)` shuffle.
    The first 10 of that shuffled order are the class's 10-shot picks; the
    first 5 / first 1 are prefixes of the same list, which is what guarantees
    1-shot subset-of 5-shot subset-of 10-shot nesting per seed.
  - Representative annotation for a (class, image) pair = first occurrence of
    that class in the image's label file (top-to-bottom line order), matching
    the same "first seen" tie-break kshot_split.py uses.
  - Only the single selected annotation for a selected image is written to
    the few-shot JSON (other boxes/classes present in that image are dropped
    from the file, even though they still exist in the source image).
  - random.Random is reseeded with the literal seed integer (1..10) per seed,
    so results are fully reproducible from this file alone.

Output: TXL-PBC/annotations_exact_instance/train_{K}shot_seed{N}.json
        (standard-COCO images/annotations/categories, same schema as
        annotations_std/*.json) plus a manifest.json documenting every pick.
val.json / test.json are NOT touched or duplicated here - existing
annotations_std/val.json and annotations_std/test.json remain the full,
unmodified val/test data for all shots and seeds.
"""
import json
import os
import random

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXL-PBC")
LABELS_DIR = os.path.join(ROOT, "labels", "train")
IMAGES_DIR = os.path.join(ROOT, "images", "train")
OUT_DIR = os.path.join(ROOT, "annotations_exact_instance")

SHOTS = [1, 5, 10]
SEEDS = list(range(1, 11))
CLASS_ORDER = [0, 1, 2]  # WBC, RBC, Platelets - fixed processing AND output order
CATEGORIES = [
    {"id": 0, "name": "WBC"},
    {"id": 1, "name": "RBC"},
    {"id": 2, "name": "Platelets"},
]
MAX_SHOT = max(SHOTS)


def yolo_to_coco_bbox(x_center, y_center, w, h, img_w, img_h):
    box_w = w * img_w
    box_h = h * img_h
    x_min = x_center * img_w - box_w / 2.0
    y_min = y_center * img_h - box_h / 2.0
    return [x_min, y_min, box_w, box_h]


def load_train_labels():
    """stem -> list of (class_id, x, y, w, h) in label-file line order."""
    index = {}
    for fname in sorted(os.listdir(LABELS_DIR)):
        stem = os.path.splitext(fname)[0]
        boxes = []
        with open(os.path.join(LABELS_DIR, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, x, y, w, h = line.split()
                boxes.append((int(cls), float(x), float(y), float(w), float(h)))
        index[stem] = boxes
    return index


def first_annotation_per_class(label_index):
    """class_id -> {stem: (x, y, w, h)} using first occurrence per image."""
    per_class = {c: {} for c in CLASS_ORDER}
    for stem, boxes in label_index.items():
        for cls, x, y, w, h in boxes:
            if stem not in per_class[cls]:
                per_class[cls][stem] = (x, y, w, h)
    return per_class


def select_seed(seed, per_class):
    """Return {class_id: [stem, ...]} with MAX_SHOT stems each, seed-shuffled,
    globally disjoint across classes (processed in CLASS_ORDER)."""
    rng = random.Random(seed)
    used_stems = set()
    picks = {}
    for cls in CLASS_ORDER:
        candidates = sorted(s for s in per_class[cls] if s not in used_stems)
        rng.shuffle(candidates)
        chosen = candidates[:MAX_SHOT]
        if len(chosen) < MAX_SHOT:
            raise RuntimeError(
                f"seed {seed}, class {cls}: only {len(chosen)} disjoint "
                f"candidate images available, need {MAX_SHOT}"
            )
        used_stems.update(chosen)
        picks[cls] = chosen
    return picks


def build_shot_json(K, seed, picks, per_class, image_size_cache):
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for cls in CLASS_ORDER:
        for stem in picks[cls][:K]:
            if stem not in image_size_cache:
                img_path = os.path.join(IMAGES_DIR, stem + ".png")
                with Image.open(img_path) as pil_img:
                    image_size_cache[stem] = pil_img.size
            img_w, img_h = image_size_cache[stem]

            images.append({
                "id": img_id,
                "file_name": stem + ".png",
                "width": img_w,
                "height": img_h,
            })

            x, y, w, h = per_class[cls][stem]
            bbox = yolo_to_coco_bbox(x, y, w, h, img_w, img_h)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            img_id += 1
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    label_index = load_train_labels()
    per_class = first_annotation_per_class(label_index)

    for cls in CLASS_ORDER:
        name = CATEGORIES[cls]["name"]
        print(f"class {cls} ({name}): {len(per_class[cls])} candidate train images")

    image_size_cache = {}
    manifest = {
        "dataset": "TXL-PBC",
        "source_split": "train",
        "policy": "exact-K-instance (CD-FSOD official policy, cross-class disjoint images)",
        "class_order": ["WBC", "RBC", "Platelets"],
        "category_ids": {"WBC": 0, "RBC": 1, "Platelets": 2},
        "shots": SHOTS,
        "seeds": SEEDS,
        "seeds_detail": {},
    }

    for seed in SEEDS:
        picks = select_seed(seed, per_class)
        manifest["seeds_detail"][str(seed)] = {
            CATEGORIES[cls]["name"]: picks[cls] for cls in CLASS_ORDER
        }
        for K in SHOTS:
            out = build_shot_json(K, seed, picks, per_class, image_size_cache)
            out_path = os.path.join(OUT_DIR, f"train_{K}shot_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            n_img = len(out["images"])
            n_ann = len(out["annotations"])
            print(f"seed {seed}, {K}-shot: {n_img} images, {n_ann} annotations -> {out_path}")
            assert n_img == 3 * K and n_ann == 3 * K

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
