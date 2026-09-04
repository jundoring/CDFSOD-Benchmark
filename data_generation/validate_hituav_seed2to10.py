import json

ROOT = "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset"
FEWSHOT_DIR = f"{ROOT}/fewshot"
TRAIN_JSON = f"{ROOT}/normal_json/annotations_std/train.json"
VAL_JSON = f"{ROOT}/normal_json/annotations_std/val.json"
TEST_JSON = f"{ROOT}/normal_json/annotations_std/test.json"
EXPECTED_CLASSES = {"Person": 0, "Car": 1, "Bicycle": 2, "OtherVehicle": 3}
SHOTS = (1, 5, 10)
SEEDS = range(2, 11)

train_files = {im["file_name"] for im in json.load(open(TRAIN_JSON))["images"]}
val_files = {im["file_name"] for im in json.load(open(VAL_JSON))["images"]}
test_files = {im["file_name"] for im in json.load(open(TEST_JSON))["images"]}

errors = []
for seed in SEEDS:
    by_shot = {}
    for K in SHOTS:
        path = f"{FEWSHOT_DIR}/train_{K}shot_seed{seed}.json"
        data = json.load(open(path))
        by_shot[K] = data

        cat_map = {c["id"]: c["name"] for c in data["categories"]}
        for name, cid in EXPECTED_CLASSES.items():
            if cat_map.get(cid) != name:
                errors.append(f"seed{seed}/{K}shot: category {cid} != {name}")

        from collections import Counter
        per_cls = Counter(a["category_id"] for a in data["annotations"])
        for name, cid in EXPECTED_CLASSES.items():
            if per_cls.get(cid, 0) != K:
                errors.append(f"seed{seed}/{K}shot: class {name} count {per_cls.get(cid,0)} != {K}")
        if len(data["images"]) != 4 * K or len(data["annotations"]) != 4 * K:
            errors.append(f"seed{seed}/{K}shot: images/annos != {4*K}")

        img_ids = {im["id"] for im in data["images"]}
        if len(img_ids) != len(data["images"]):
            errors.append(f"seed{seed}/{K}shot: duplicate image id")
        ann_ids = [a["id"] for a in data["annotations"]]
        if len(set(ann_ids)) != len(ann_ids):
            errors.append(f"seed{seed}/{K}shot: duplicate annotation id")

        img_wh = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            iw, ih = img_wh[a["image_id"]]
            if w <= 0 or h <= 0:
                errors.append(f"seed{seed}/{K}shot: ann {a['id']} non-positive bbox {a['bbox']}")
            if x < -1e-2 or y < -1e-2 or x + w > iw + 1e-2 or y + h > ih + 1e-2:
                errors.append(f"seed{seed}/{K}shot: ann {a['id']} bbox out of bounds {a['bbox']} vs {iw}x{ih}")

        support_files = {im["file_name"] for im in data["images"]}
        not_in_train = support_files - train_files
        leak_val = support_files & val_files
        leak_test = support_files & test_files
        if not_in_train:
            errors.append(f"seed{seed}/{K}shot: {len(not_in_train)} images not in train")
        if leak_val:
            errors.append(f"seed{seed}/{K}shot: {len(leak_val)} images leak into val")
        if leak_test:
            errors.append(f"seed{seed}/{K}shot: {len(leak_test)} images leak into test")

        # duplicate file_name within file (cross-class disjoint check)
        fnames = [im["file_name"] for im in data["images"]]
        if len(set(fnames)) != len(fnames):
            errors.append(f"seed{seed}/{K}shot: duplicate file_name (image reused across classes)")

    # nesting check
    for small, big in ((1, 5), (5, 10)):
        small_pairs = {(im["file_name"], a["category_id"])
                        for im, a in zip(sorted(by_shot[small]["images"], key=lambda x: x["id"]),
                                          sorted(by_shot[small]["annotations"], key=lambda x: x["image_id"]))}
        big_pairs = {(im["file_name"], a["category_id"])
                     for im, a in zip(sorted(by_shot[big]["images"], key=lambda x: x["id"]),
                                       sorted(by_shot[big]["annotations"], key=lambda x: x["image_id"]))}
        if not small_pairs.issubset(big_pairs):
            errors.append(f"seed{seed}: {small}-shot NOT subset of {big}-shot")

print(f"Checked {len(list(SEEDS))} seeds x {len(SHOTS)} shots = {len(list(SEEDS))*len(SHOTS)} files")
if errors:
    print(f"=== {len(errors)} ERROR(S) ===")
    for e in errors:
        print("FAIL:", e)
else:
    print("ALL 27 HIT-UAV seed2-10 FILES PASSED VALIDATION")
