"""TXL-PBC 1-shot seed1 smoke test (Stage 5) - official FSOD-VFM pipeline,
official CD-FSOD hyperparameters, run against a 5-image derived test subset
(not the full 126-image test.json). Adds instrumentation and support-crop /
prediction visualizations around the official support_util.py / metric.py
functions - no official code is modified.
"""
import json
import os
import time

import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import pycocotools.coco

import model.dinov2
import model.sam2
import support_util
import query_util
import metric
from chatrex.upn import UPNWrapper

DATA_DIR = "/home/hjsjune/workspace/Dataset/TXL-PBC_Dataset/TXL-PBC"
SUPPORT_JSON = "/home/hjsjune/workspace/FSOD-VFM/data/TXL-PBC/exact_instance/1shot_seed1.json"
TEST_JSON = "/home/hjsjune/workspace/FSOD-VFM/data/TXL-PBC/exact_instance/smoke_test_5to10img.json"
TEST_IMG_DIR = os.path.join(DATA_DIR, "images", "test")
OUT_DIR = "/home/hjsjune/workspace/FSOD-VFM/output/smoke_test_TXL-PBC_1shot_seed1"
SUPPORT_CROP_DIR = os.path.join(OUT_DIR, "support_crops")
PRED_VIZ_DIR = os.path.join(OUT_DIR, "prediction_viz")

TARGET_CATEGORIES = ["WBC", "RBC", "Platelets"]
MIN_THRESHOLD = 0.01
DIFFUSION_STEPS = 30
ALP = 0.3
LAMB = 0.5
POINTS_PER_SIDE = 32
DINOV2_IMAGE_SIZE = 630
DEVICE = "cuda"

os.makedirs(SUPPORT_CROP_DIR, exist_ok=True)
os.makedirs(PRED_VIZ_DIR, exist_ok=True)

log = {"steps": []}


def record(step, **kwargs):
    entry = {"step": step, **kwargs}
    log["steps"].append(entry)
    print(f"[{step}]", kwargs)


torch.cuda.reset_peak_memory_stats()

# ---- load models (official loaders, no modification) ----
t0 = time.time()
upn = UPNWrapper("./checkpoints/upn_large.pth")
feat_extractor, image_transform = model.dinov2.load_dinov2_model(
    DEVICE, "dinov2_vitl14", image_size=(DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE),
    repo_or_dir="./dinov2", pretrained="./checkpoints/dinov2_vitl14_pretrain.pth",
)
sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
    model_type="large", device=DEVICE, points_per_side=POINTS_PER_SIDE,
)
torch.cuda.synchronize()
record("models_loaded", duration_s=time.time() - t0,
       allocated_mib=torch.cuda.memory_allocated() / 1024**2,
       reserved_mib=torch.cuda.memory_reserved() / 1024**2)

# ---- load support set (official format) ----
with open(SUPPORT_JSON) as f:
    support_data = json.load(f)

per_class_support_count = {cls: len(v) for cls, v in support_data.items()}
record("support_loaded", per_class_count=per_class_support_count,
       classes=list(support_data.keys()))
assert per_class_support_count == {"WBC": 1, "RBC": 1, "Platelets": 1}, \
    f"expected 1 instance/class for 1-shot, got {per_class_support_count}"

# ---- save support crop visualizations (extra instrumentation, not official code) ----
for cls, samples in support_data.items():
    for i, sample in enumerate(samples):
        img_path = os.path.join(DATA_DIR, sample["image"])
        pil_img = PIL.Image.open(img_path).convert("RGB")
        x, y, w, h = sample["bbox"]
        # full image with bbox drawn
        annotated = pil_img.copy()
        draw = PIL.ImageDraw.Draw(annotated)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        annotated.save(os.path.join(SUPPORT_CROP_DIR, f"{cls}_{i}_full_with_bbox.png"))
        # tight crop of just the selected bbox region
        crop = pil_img.crop((max(0, x), max(0, y), x + w, y + h))
        crop.save(os.path.join(SUPPORT_CROP_DIR, f"{cls}_{i}_crop.png"))
record("support_crops_saved", dir=SUPPORT_CROP_DIR, n_files=len(os.listdir(SUPPORT_CROP_DIR)))

# ---- build memory bank / prototypes (official support_util functions, unmodified) ----
t0 = time.time()
memory_bank = support_util.extract_support_features(
    support_data, sam2_predictor, "DINOV2", feat_extractor, image_transform, DATA_DIR, DEVICE,
)
proto_feat, proto_cls = support_util.compute_prototype_weights(memory_bank, DEVICE)
record("prototypes_built", duration_s=time.time() - t0, proto_classes=proto_cls,
       proto_feat_shape=list(proto_feat.shape))
assert set(proto_cls) == set(TARGET_CATEGORIES), f"proto_cls {proto_cls} != {TARGET_CATEGORIES}"
assert not torch.isnan(proto_feat).any().item(), "NaN in proto_feat"
assert not torch.isinf(proto_feat).any().item(), "Inf in proto_feat"

# ---- load smoke-test query set (5-image derived subset, official loader) ----
image_paths, coco_style_loader = query_util.load_voc2007_coco_json(TEST_JSON, TEST_IMG_DIR)
record("query_set_loaded", n_images=len(image_paths), test_json=TEST_JSON, test_img_dir=TEST_IMG_DIR)

# ---- run inference (official metric.generate_coco_style_predictions_upn, unmodified) ----
t0 = time.time()
results = metric.generate_coco_style_predictions_upn(
    coco_style_loader, TEST_IMG_DIR, sam2_predictor, "DINOV2", feat_extractor,
    image_transform, proto_feat, proto_cls, upn, DIFFUSION_STEPS, ALP, LAMB, DEVICE, MIN_THRESHOLD,
)
torch.cuda.synchronize()
inference_duration = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

n_images = len(image_paths)
per_image_time = inference_duration / n_images if n_images else float("nan")
images_with_pred = len({r["image_id"] for r in results})
empty_pred_ratio = 1 - (images_with_pred / n_images) if n_images else float("nan")

record("inference_done", duration_s=inference_duration, per_image_s=per_image_time,
       n_predictions=len(results), images_with_predictions=images_with_pred,
       n_images=n_images, empty_prediction_ratio=empty_pred_ratio,
       peak_vram_mib=peak_vram)

# ---- checks: bbox within image bounds, category mapping, NaN/Inf ----
id_to_name = metric.get_category_id_to_name(coco_style_loader)
img_wh = {im["id"]: (im["width"], im["height"]) for im in coco_style_loader.dataset["images"]}
bbox_out_of_bounds = 0
nan_or_inf_scores = 0
category_ids_seen = set()
for r in results:
    x, y, w, h = r["bbox"]
    iw, ih = img_wh[r["image_id"]]
    if x < -1e-2 or y < -1e-2 or x + w > iw + 1e-2 or y + h > ih + 1e-2:
        bbox_out_of_bounds += 1
    s = r["score"]
    if s != s or s in (float("inf"), float("-inf")):
        nan_or_inf_scores += 1
    category_ids_seen.add(r["category_id"])

record("prediction_sanity", bbox_out_of_bounds=bbox_out_of_bounds,
       nan_or_inf_scores=nan_or_inf_scores,
       category_ids_seen={cid: id_to_name[cid] for cid in category_ids_seen},
       category_mapping_valid=category_ids_seen.issubset(set(id_to_name.keys())))

# ---- save prediction visualizations (extra instrumentation, not official code) ----
preds_by_image = {}
for r in results:
    preds_by_image.setdefault(r["image_id"], []).append(r)

for img_dict in coco_style_loader.dataset["images"]:
    img_id = img_dict["id"]
    img_path = os.path.join(TEST_IMG_DIR, img_dict["file_name"])
    pil_img = PIL.Image.open(img_path).convert("RGB")
    draw = PIL.ImageDraw.Draw(pil_img)
    for r in preds_by_image.get(img_id, []):
        x, y, w, h = r["bbox"]
        cls_name = id_to_name.get(r["category_id"], "?")
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
        draw.text((x, max(0, y - 10)), f"{cls_name}:{r['score']:.2f}", fill="lime")
    pil_img.save(os.path.join(PRED_VIZ_DIR, f"{img_dict['file_name']}"))
record("prediction_viz_saved", dir=PRED_VIZ_DIR, n_files=len(os.listdir(PRED_VIZ_DIR)))

# ---- run official evaluator (metric.run_coco_eval, unmodified) ----
pred_json_path = os.path.join(OUT_DIR, "smoke_test_predictions.json")
os.makedirs("./results", exist_ok=True)
eval_results = metric.run_coco_eval(
    TEST_JSON, results, pred_json=pred_json_path,
    target_categories=TARGET_CATEGORIES, filter_by_categories=True,
)
record("evaluator_done", eval_stats=eval_results.get("bbox", {}).get("stats"))

with open(os.path.join(OUT_DIR, "smoke_test_log.json"), "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nSMOKE_TEST_COMPLETE_OK")
print(f"peak_vram_mib={peak_vram:.1f}")
print(f"per_image_inference_s={per_image_time:.3f}")
print(f"empty_prediction_ratio={empty_pred_ratio:.3f}")
print(f"bbox_out_of_bounds={bbox_out_of_bounds}")
print(f"nan_or_inf_scores={nan_or_inf_scores}")
