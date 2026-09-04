"""Validation report for TXL-PBC/annotations_exact_instance/*.json.

Checks (per user's requirements):
  1. per seed/shot: image count and annotation count
  2. per-class annotation counts and totals
  3. 1-shot subset of 5-shot subset of 10-shot (per seed)
  4. every selected image_id (file_name) exists only in the original train split
  5. no image leakage between train/val/test
  6. category id/order correctness (0=WBC,1=RBC,2=Platelets)
  7. bbox/area validity (positive size, within image bounds)
  8. SHA-256 of original train/val/test + existing K-image files unchanged
"""
import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXL-PBC")
EXACT_DIR = os.path.join(ROOT, "annotations_exact_instance")
SHOTS = [1, 5, 10]
SEEDS = list(range(1, 11))
CATEGORIES_EXPECTED = [
    {"id": 0, "name": "WBC"},
    {"id": 1, "name": "RBC"},
    {"id": 2, "name": "Platelets"},
]

PROTECTED_FILES = []
for sub in ("annotations_std", "fewshot"):
    d = os.path.join(ROOT, sub)
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            PROTECTED_FILES.append(os.path.join(sub, f))
for K in (1, 5, 10):
    d = os.path.join(ROOT, "support_sets", f"{K}shot")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            PROTECTED_FILES.append(os.path.join("support_sets", f"{K}shot", f))
PROTECTED_FILES.append(os.path.join("support_sets", "manifest.json"))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_baseline_hashes(path):
    baseline = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, fname = line.split(maxsplit=1)
            baseline[fname.strip()] = digest
    return baseline


def main():
    errors = []
    report_lines = []

    train_files = set(os.listdir(os.path.join(ROOT, "images", "train")))
    val_files = set(os.listdir(os.path.join(ROOT, "images", "val")))
    test_files = set(os.listdir(os.path.join(ROOT, "images", "test")))

    report_lines.append("## 1-2. Per seed/shot image & annotation counts\n")
    report_lines.append("| seed | shot | images | annotations | WBC | RBC | Platelets |")
    report_lines.append("|---|---|---|---|---|---|---|")

    all_data = {}  # (seed, K) -> data
    for seed in SEEDS:
        for K in SHOTS:
            path = os.path.join(EXACT_DIR, f"train_{K}shot_seed{seed}.json")
            with open(path) as f:
                data = json.load(f)
            all_data[(seed, K)] = data

            n_img = len(data["images"])
            n_ann = len(data["annotations"])
            cls_count = {0: 0, 1: 0, 2: 0}
            for ann in data["annotations"]:
                cls_count[ann["category_id"]] += 1

            if n_img != 3 * K:
                errors.append(f"seed{seed}/{K}shot: image count {n_img} != {3*K}")
            if n_ann != 3 * K:
                errors.append(f"seed{seed}/{K}shot: annotation count {n_ann} != {3*K}")
            for c in (0, 1, 2):
                if cls_count[c] != K:
                    errors.append(f"seed{seed}/{K}shot: class {c} count {cls_count[c]} != {K}")
            if n_img != n_ann:
                errors.append(f"seed{seed}/{K}shot: images ({n_img}) != annotations ({n_ann}), i.e. not 1 ann/image")

            report_lines.append(
                f"| {seed} | {K} | {n_img} | {n_ann} | {cls_count[0]} | {cls_count[1]} | {cls_count[2]} |"
            )

            # category id/order correctness
            if data["categories"] != CATEGORIES_EXPECTED:
                errors.append(f"seed{seed}/{K}shot: categories mismatch: {data['categories']}")

            # image_id uniqueness = one annotation per image
            img_ids_in_images = [im["id"] for im in data["images"]]
            if len(set(img_ids_in_images)) != len(img_ids_in_images):
                errors.append(f"seed{seed}/{K}shot: duplicate image id in images[]")
            ann_img_ids = [a["image_id"] for a in data["annotations"]]
            if len(set(ann_img_ids)) != len(ann_img_ids):
                errors.append(f"seed{seed}/{K}shot: duplicate image_id across annotations (>1 ann per image)")
            if set(ann_img_ids) != set(img_ids_in_images):
                errors.append(f"seed{seed}/{K}shot: annotation image_id set != images id set")

            # file_name (source image) uniqueness across the whole shot file
            file_names = [im["file_name"] for im in data["images"]]
            if len(set(file_names)) != len(file_names):
                errors.append(f"seed{seed}/{K}shot: duplicate file_name (image reused across classes)")

            # train-only membership, no leakage
            for fn in file_names:
                if fn not in train_files:
                    errors.append(f"seed{seed}/{K}shot: {fn} not found in images/train")
                if fn in val_files:
                    errors.append(f"seed{seed}/{K}shot: LEAKAGE - {fn} also present in images/val")
                if fn in test_files:
                    errors.append(f"seed{seed}/{K}shot: LEAKAGE - {fn} also present in images/test")

            # bbox / area validity
            img_wh = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
            for ann in data["annotations"]:
                x, y, w, h = ann["bbox"]
                img_w, img_h = img_wh[ann["image_id"]]
                if w <= 0 or h <= 0:
                    errors.append(f"seed{seed}/{K}shot: ann {ann['id']} non-positive bbox size {ann['bbox']}")
                # tolerance matches the sub-pixel float noise already present in the
                # existing annotations_std/*.json (same YOLO->COCO conversion formula,
                # max observed overshoot ~3e-4 px on edge-touching boxes)
                BBOX_TOL = 1e-2
                if x < -BBOX_TOL or y < -BBOX_TOL or x + w > img_w + BBOX_TOL or y + h > img_h + BBOX_TOL:
                    errors.append(
                        f"seed{seed}/{K}shot: ann {ann['id']} bbox out of image bounds "
                        f"{ann['bbox']} vs image {img_w}x{img_h}"
                    )
                expected_area = w * h
                if abs(ann["area"] - expected_area) > 1e-3:
                    errors.append(f"seed{seed}/{K}shot: ann {ann['id']} area mismatch {ann['area']} vs {expected_area}")

    report_lines.append("\n## 3. Nesting check (1-shot subset of 5-shot subset of 10-shot, per seed)\n")
    report_lines.append("| seed | 1-shot ⊂ 5-shot | 5-shot ⊂ 10-shot |")
    report_lines.append("|---|---|---|")
    for seed in SEEDS:
        fn1 = {im["file_name"] for im in all_data[(seed, 1)]["images"]}
        fn5 = {im["file_name"] for im in all_data[(seed, 5)]["images"]}
        fn10 = {im["file_name"] for im in all_data[(seed, 10)]["images"]}
        ok_1_5 = fn1.issubset(fn5)
        ok_5_10 = fn5.issubset(fn10)
        if not ok_1_5:
            errors.append(f"seed{seed}: 1-shot NOT subset of 5-shot")
        if not ok_5_10:
            errors.append(f"seed{seed}: 5-shot NOT subset of 10-shot")

        # also check per-class nesting (annotation-level, not just image set)
        for K_small, K_big in ((1, 5), (5, 10)):
            small = all_data[(seed, K_small)]
            big = all_data[(seed, K_big)]
            small_by_cls = {}
            for ann, im in zip(small["annotations"], small["images"]):
                pass
            small_fn_cat = {(im["file_name"], a["category_id"])
                             for im, a in zip(
                                 sorted(small["images"], key=lambda x: x["id"]),
                                 sorted(small["annotations"], key=lambda x: x["image_id"]))}
            big_fn_cat = {(im["file_name"], a["category_id"])
                          for im, a in zip(
                              sorted(big["images"], key=lambda x: x["id"]),
                              sorted(big["annotations"], key=lambda x: x["image_id"]))}
            if not small_fn_cat.issubset(big_fn_cat):
                errors.append(f"seed{seed}: {K_small}-shot per-class picks NOT subset of {K_big}-shot")

        report_lines.append(f"| {seed} | {'OK' if ok_1_5 else 'FAIL'} | {ok_5_10 and 'OK' or 'FAIL'} |")

    report_lines.append("\n## 4-5. Train-only membership / leakage\n")
    all_selected_files = set()
    for (seed, K), data in all_data.items():
        all_selected_files.update(im["file_name"] for im in data["images"])
    leak_val = all_selected_files & val_files
    leak_test = all_selected_files & test_files
    not_in_train = all_selected_files - train_files
    report_lines.append(f"- total distinct selected images across all seeds/shots: {len(all_selected_files)}")
    report_lines.append(f"- overlap with images/val: {len(leak_val)} (expect 0)")
    report_lines.append(f"- overlap with images/test: {len(leak_test)} (expect 0)")
    report_lines.append(f"- selected images NOT in images/train: {len(not_in_train)} (expect 0)")
    if leak_val or leak_test or not_in_train:
        errors.append("train/val/test leakage or out-of-train selection detected (see counts above)")

    report_lines.append("\n## 8. SHA-256 unchanged check for original train/val/test + existing K-image files\n")
    baseline_path = "/tmp/claude-1004/-home-hjsjune-workspace-FT-FSOD/3f1bf8ed-1879-4046-a390-d088b7dc76bd/scratchpad/baseline_hashes_before.txt"
    baseline = load_baseline_hashes(baseline_path)
    report_lines.append("| file | unchanged |")
    report_lines.append("|---|---|")
    for rel in PROTECTED_FILES:
        full = os.path.join(ROOT, rel)
        cur = sha256_of(full)
        base = baseline.get(rel)
        ok = (base == cur)
        if not ok:
            errors.append(f"PROTECTED FILE CHANGED: {rel} (baseline {base}, current {cur})")
        report_lines.append(f"| {rel} | {'OK' if ok else 'CHANGED!'} |")
    missing_in_baseline = set(PROTECTED_FILES) - set(baseline.keys())
    if missing_in_baseline:
        errors.append(f"files missing from baseline hash list: {missing_in_baseline}")

    report_lines.append("\n## 9. train/labels source images/labels also unchanged (spot check via directory listing count)\n")
    n_train_img = len(train_files)
    n_val_img = len(val_files)
    n_test_img = len(test_files)
    report_lines.append(f"- images/train count: {n_train_img} (expect 882)")
    report_lines.append(f"- images/val count: {n_val_img} (expect 252)")
    report_lines.append(f"- images/test count: {n_test_img} (expect 126)")
    if n_train_img != 882 or n_val_img != 252 or n_test_img != 126:
        errors.append("train/val/test image counts changed from expected baseline")

    print("\n".join(report_lines))
    print("\n\n=== ERRORS ===")
    if errors:
        for e in errors:
            print("FAIL:", e)
        print(f"\n{len(errors)} error(s) found.")
    else:
        print("No errors found. All checks passed.")

    return errors


if __name__ == "__main__":
    errs = main()
    raise SystemExit(1 if errs else 0)
