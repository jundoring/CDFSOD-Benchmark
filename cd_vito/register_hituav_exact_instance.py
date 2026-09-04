"""Detectron2/CD-ViTO dataset registration for HIT-UAV exact-K-instance
support sets, all seeds (1..10).

Uses the canonical HIT-UAV dataset root directly (Dataset/
HIT-UAV-Infrared-Thermal-Dataset, shared with FT-FSOD via its own
data/HIT-UAV symlink). Additive to, and separate from,
register_txlpbc_hituav.py's hituav_train_{K}shot_seed1 name (same seed1
file, registered again here under a uniform hituav_exact_* name so seed1..10
can be looped over identically) - nothing here removes or overwrites that
name, or hituav_val/hituav_test (reused as-is, DontCare already excluded at
the annotations_std conversion step upstream).

Registered names:
  hituav_exact_train_{1,5,10}shot_seed{1..10}  <- fewshot/train_{K}shot_seed{N}.json
Test: reuse the existing "hituav_test" registration (normal_json/
annotations_std/test.json) - no separate "hituav_exact_test" needed since
val/test are full, unmodified data under both naming schemes.
"""
import glob
import json
import os
import re

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

HITUAV_ROOT = "/home/hjsjune/workspace/Dataset/HIT-UAV-Infrared-Thermal-Dataset"
IMAGE_ROOT = os.path.join(HITUAV_ROOT, "normal_json", "train")
FEWSHOT_DIR = os.path.join(HITUAV_ROOT, "fewshot")

_FNAME_RE = re.compile(r"train_(\d+)shot_seed(\d+)\.json$")


def _register(name, json_file, image_root):
    if name in DatasetCatalog.list():
        return
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    classes = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
    register_coco_instances(name, {}, json_file, image_root)
    MetadataCatalog.get(name).set(thing_classes=classes)


def register_hituav_exact_instance():
    if not os.path.isdir(FEWSHOT_DIR):
        return
    for json_path in sorted(glob.glob(os.path.join(FEWSHOT_DIR, "train_*shot_seed*.json"))):
        m = _FNAME_RE.search(os.path.basename(json_path))
        if not m:
            continue
        K, seed = m.group(1), m.group(2)
        name = f"hituav_exact_train_{K}shot_seed{seed}"
        _register(name, json_path, IMAGE_ROOT)


register_hituav_exact_instance()
