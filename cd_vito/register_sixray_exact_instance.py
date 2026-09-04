"""Detectron2/CD-ViTO dataset registration for the fair-conditions SIXray-D
exact-K-instance re-run, all seeds (1..10).

Uses the CANONICAL SIXray-D dataset root directly
(Dataset/SIXray-D/SIXray-D, shared with FT-FSOD via its own data/SIXray-D
symlink) - NOT the separate, non-canonical CDFSOD-benchmark/datasets/SIXRAY
copy (different image set, and its {K}_shot.json files were generated with
seed=None, i.e. non-reproducible and not the validated exact-instance
support set).

This is ADDITIVE to, and separate from, register_sixray_seed_shots() in
detectron2/data/datasets/builtin.py (which registers SIXRAY_{k}shot_seed{N}
from the non-canonical copy) - nothing here touches those names or files.

Registered names:
  sixray_exact_train_{1,5,10}shot_seed{1..10}  <- fewshot/train_{K}shot_seed{N}.json
  sixray_exact_val                              <- val.json (1005 images)
  sixray_exact_test                             <- test.json (836 images)
(train/val/test are three separate, non-overlapping files/image sets.)
"""
import glob
import json
import os
import re

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

SIXRAY_D_ROOT = "/home/hjsjune/workspace/Dataset/SIXray-D/SIXray-D"
IMAGE_ROOT = os.path.join(SIXRAY_D_ROOT, "images")
FEWSHOT_DIR = os.path.join(SIXRAY_D_ROOT, "fewshot")

_FNAME_RE = re.compile(r"train_(\d+)shot_seed(\d+)\.json$")


def _register(name, json_file, image_root):
    if name in DatasetCatalog.list():
        return
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    classes = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
    register_coco_instances(name, {}, json_file, image_root)
    MetadataCatalog.get(name).set(thing_classes=classes)


def register_sixray_exact_instance():
    if not os.path.isdir(SIXRAY_D_ROOT):
        return
    for json_path in sorted(glob.glob(os.path.join(FEWSHOT_DIR, "train_*shot_seed*.json"))):
        m = _FNAME_RE.search(os.path.basename(json_path))
        if not m:
            continue
        K, seed = m.group(1), m.group(2)
        name = f"sixray_exact_train_{K}shot_seed{seed}"
        _register(name, json_path, IMAGE_ROOT)
    _register("sixray_exact_val", os.path.join(SIXRAY_D_ROOT, "val.json"), IMAGE_ROOT)
    _register("sixray_exact_test", os.path.join(SIXRAY_D_ROOT, "test.json"), IMAGE_ROOT)


register_sixray_exact_instance()
