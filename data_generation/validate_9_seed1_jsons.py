import json
import hashlib

DATASETS = {
    "TXL-PBC": {
        "fewshot_dir": "/home/hjsjune/workspace/Dataset/TXL-PBC_Dataset/TXL-PBC/annotations_exact_instance",
        "fname_pattern": "train_{K}shot_seed1.json",
        "test_json": "/home/hjsjune/workspace/Dataset/TXL-PBC_Dataset/TXL-PBC/annotations_std/test.json",
        "expected_classes": {"WBC": 0, "RBC": 1, "Platelets": 2},
    },
    "SIXray-D": {
        "fewshot_dir": "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D/fewshot",
        "fname_pattern": "train_{K}shot_seed1.json",
        "test_json": "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D/test.json",
        "expected_classes": {"Gun": 1, "Knife": 2, "Wrench": 3, "Pliers": 4, "Scissors": 5},
    },
    "HIT-UAV": {
        "fewshot_dir": "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset/fewshot",
        "fname_pattern": "train_{K}shot_seed1.json",
        "test_json": "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset/normal_json/annotations_std/test.json",
        "expected_classes": {"Person": 0, "Car": 1, "Bicycle": 2, "OtherVehicle": 3},
    },
}
SHOTS = (1, 5, 10)

errors = []
all_ok = True

for dsname, cfg in DATASETS.items():
    test_data = json.load(open(cfg["test_json"]))
    test_files = {im["file_name"] for im in test_data["images"]}

    by_shot = {}
    for K in SHOTS:
        path = f"{cfg['fewshot_dir']}/{cfg['fname_pattern'].format(K=K)}"
        data = json.load(open(path))
        by_shot[K] = data
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()

        n_classes = len(cfg["expected_classes"])
        expected_total = n_classes * K

        # category mapping check
        cat_map = {c["id"]: c["name"] for c in data["categories"]}
        for name, cid in cfg["expected_classes"].items():
            if cat_map.get(cid) != name:
                errors.append(f"{dsname}/{K}shot: category id {cid} expected '{name}', got '{cat_map.get(cid)}'")

        # class-wise annotation count
        from collections import Counter
        per_cls = Counter(a["category_id"] for a in data["annotations"])
        for name, cid in cfg["expected_classes"].items():
            if per_cls.get(cid, 0) != K:
                errors.append(f"{dsname}/{K}shot: class '{name}' has {per_cls.get(cid,0)} annos, expected {K}")

        # total count
        if len(data["annotations"]) != expected_total:
            errors.append(f"{dsname}/{K}shot: total annos {len(data['annotations'])} != expected {expected_total}")

        # image_id / annotation_id validity
        img_ids = {im["id"] for im in data["images"]}
        if len(img_ids) != len(data["images"]):
            errors.append(f"{dsname}/{K}shot: duplicate image id")
        ann_ids = [a["id"] for a in data["annotations"]]
        if len(set(ann_ids)) != len(ann_ids):
            errors.append(f"{dsname}/{K}shot: duplicate annotation id")
        for a in data["annotations"]:
            if a["image_id"] not in img_ids:
                errors.append(f"{dsname}/{K}shot: annotation {a['id']} references missing image_id {a['image_id']}")

        # bbox validity
        img_wh = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            iw, ih = img_wh.get(a["image_id"], (None, None))
            if w <= 0 or h <= 0:
                errors.append(f"{dsname}/{K}shot: ann {a['id']} non-positive bbox size {a['bbox']}")
            if iw is not None:
                tol = 1e-2
                if x < -tol or y < -tol or x + w > iw + tol or y + h > ih + tol:
                    errors.append(f"{dsname}/{K}shot: ann {a['id']} bbox out of bounds {a['bbox']} vs {iw}x{ih}")

        # support/test leakage
        support_files = {im["file_name"] for im in data["images"]}
        leak = support_files & test_files
        if leak:
            errors.append(f"{dsname}/{K}shot: {len(leak)} support images overlap with test set: {list(leak)[:5]}")

        print(f"{dsname}/{K}shot: images={len(data['images'])} annos={len(data['annotations'])} "
              f"per_class={dict(per_cls)} sha256={sha[:16]}...")

    # nesting check (annotation-id based, and (file_name, category_id) based)
    for small, big in ((1, 5), (5, 10)):
        small_pairs = {(next(im['file_name'] for im in by_shot[small]['images'] if im['id']==a['image_id']), a['category_id'])
                       for a in by_shot[small]['annotations']}
        big_pairs = {(next(im['file_name'] for im in by_shot[big]['images'] if im['id']==a['image_id']), a['category_id'])
                     for a in by_shot[big]['annotations']}
        if not small_pairs.issubset(big_pairs):
            errors.append(f"{dsname}: {small}-shot NOT subset of {big}-shot (by file_name+category)")
        else:
            print(f"{dsname}: {small}-shot subset of {big}-shot -> OK")

print()
if errors:
    print(f"=== {len(errors)} ERROR(S) ===")
    for e in errors:
        print("FAIL:", e)
else:
    print("=== ALL 9 SEED1 FEW-SHOT JSONs PASSED VALIDATION ===")
