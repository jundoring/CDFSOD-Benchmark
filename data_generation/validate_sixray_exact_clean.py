import hashlib
import json
import os
import re

SIXRAY_ROOT = "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D"
TRAIN_JSON = {K: os.path.join(SIXRAY_ROOT, "fewshot", f"train_{K}shot_seed1.json") for K in (1, 5, 10)}
VAL_JSON = os.path.join(SIXRAY_ROOT, "val.json")
TEST_JSON = os.path.join(SIXRAY_ROOT, "test.json")
EXPECTED_CLASS_ORDER = ["Gun", "Knife", "Wrench", "Pliers", "Scissors"]
EXPECTED_IDS = {"Gun": 1, "Knife": 2, "Wrench": 3, "Pliers": 4, "Scissors": 5}

FT_CONFIG_DIR = "/home/hjsjune/workspace/FT-FSOD/configs_cdfsod/final_configs_sixray_exact_seed1_clean"
CD_CONFIG_DIR = "/home/hjsjune/workspace/CDFSOD-benchmark/configs/sixray_exact_instance_seed1_clean"

errors = []


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== 1. train/val/test JSON identity (path + SHA-256) ===")
for K, path in TRAIN_JSON.items():
    print(f"train {K}shot: {path}\n  sha256={sha256_of(path)}")
print(f"val:  {VAL_JSON}\n  sha256={sha256_of(VAL_JSON)}")
print(f"test: {TEST_JSON}\n  sha256={sha256_of(TEST_JSON)}")

print("\n=== 2. class order / exact-K-instance counts ===")
for K, path in TRAIN_JSON.items():
    data = json.load(open(path))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    order_ok = [cats[i] for i in sorted(cats)] == EXPECTED_CLASS_ORDER
    if not order_ok:
        errors.append(f"{K}shot: category order {[cats[i] for i in sorted(cats)]} != {EXPECTED_CLASS_ORDER}")
    from collections import Counter
    per_cls = Counter(a["category_id"] for a in data["annotations"])
    for name, cid in EXPECTED_IDS.items():
        if per_cls.get(cid, 0) != K:
            errors.append(f"{K}shot: class '{name}' has {per_cls.get(cid,0)} annos, expected {K}")
    print(f"{K}shot: order_ok={order_ok} per_class={ {cats[c]: n for c, n in per_cls.items()} }")

print("\n=== 3. val/test image overlap ===")
val_data = json.load(open(VAL_JSON))
test_data = json.load(open(TEST_JSON))
val_files = {im["file_name"] for im in val_data["images"]}
test_files = {im["file_name"] for im in test_data["images"]}
overlap = val_files & test_files
print(f"val images={len(val_files)} test images={len(test_files)} overlap={len(overlap)}")
if overlap:
    errors.append(f"val/test overlap: {len(overlap)} shared images")

print("\n=== 4. support/test image overlap (per shot) ===")
for K, path in TRAIN_JSON.items():
    data = json.load(open(path))
    support_files = {im["file_name"] for im in data["images"]}
    leak_test = support_files & test_files
    leak_val = support_files & val_files
    if leak_test:
        errors.append(f"{K}shot: {len(leak_test)} support images overlap with test.json")
    if leak_val:
        errors.append(f"{K}shot: {len(leak_val)} support images overlap with val.json")
    print(f"{K}shot: support={len(support_files)} overlap_with_val={len(leak_val)} overlap_with_test={len(leak_test)}")

print("\n=== 5. FT-FSOD config ann_file checks ===")
for K in (1, 5, 10):
    path = os.path.join(FT_CONFIG_DIR, f"grounding_dino_swin-b_finetune_SIXray-D_{K}shot_seed1.py")
    text = open(path).read()
    train_ok = f"ann_file=f'fewshot/train_{K}shot_seed1.json'" in text
    val_ok = "ann_file=f'val.json'" in text and text.count("ann_file=f'val.json'") == 1
    test_ok = text.count("ann_file=f'test.json'") == 1
    val_evaluator_ok = "ann_file=f'{data_root}/val.json'" in text
    test_evaluator_ok = "ann_file=f'{data_root}/test.json'" in text
    ok = train_ok and val_ok and test_ok and val_evaluator_ok and test_evaluator_ok
    print(f"{path}: train_ok={train_ok} val_ok={val_ok} test_ok={test_ok} "
          f"val_evaluator_ok={val_evaluator_ok} test_evaluator_ok={test_evaluator_ok}")
    if not ok:
        errors.append(f"FT-FSOD config {K}shot: ann_file check failed")

print("\n=== 6. CD-ViTO config DATASETS checks ===")
for K in (1, 5, 10):
    path = os.path.join(CD_CONFIG_DIR, f"vitl_shot{K}_sixray_finetune_seed1.yaml")
    text = open(path).read()
    train_ok = f'TRAIN: ("sixray_exact_train_{K}shot_seed1",)' in text
    test_ok = 'TEST: ("sixray_exact_test",)' in text
    proto_ok = f"prototypes_init/sixray_exact_train_{K}shot_seed1.vitl14.bbox.p{K}.sk.pkl" in text
    ok = train_ok and test_ok and proto_ok
    print(f"{path}: train_ok={train_ok} test_ok={test_ok} proto_ok={proto_ok}")
    if not ok:
        errors.append(f"CD-ViTO config {K}shot: DATASETS/prototype check failed")

print("\n=== 7. existing SIXray configs/results untouched (spot check mtimes unaffected is not reliable; check content hash of a known existing config) ===")
existing_cfg = "/home/hjsjune/workspace/FT-FSOD/configs_cdfsod/final_configs_sixray/grounding_dino_swin-b_finetune_SIXray-D_1shot.py"
print(f"{existing_cfg} sha256={sha256_of(existing_cfg)} (compare manually to earlier baseline if needed)")

print("\n\n=== SUMMARY ===")
if errors:
    for e in errors:
        print("FAIL:", e)
    print(f"\n{len(errors)} error(s).")
else:
    print("ALL CHECKS PASSED.")
