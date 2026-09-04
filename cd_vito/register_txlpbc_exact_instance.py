"""Detectron2/CD-ViTO dataset registration for the TXL-PBC exact-K-instance
few-shot policy (FT-FSOD counterpart change - see Dataset/TXL-PBC_Dataset/
TXL-PBC/annotations_exact_instance/build_txlpbc_exact_kinstance.py for how
these JSONs were generated).

This is ADDITIVE to, and completely separate from, register_txlpbc_hituav.py
(the existing K-image policy registration, which is untouched and still
registers txlpbc_train_{K}shot_seed1 / txlpbc_val / txlpbc_test from
annotations_std/). Nothing here overwrites or removes those names.

Source: Dataset/TXL-PBC_Dataset/TXL-PBC/annotations_exact_instance/
        train_{K}shot_seed{N}.json  (K in {1,5,10}, N in {1..10})
Registered train names: txlpbc_exact_train_{K}shot_seed{N}
Val/test: reuses the already-registered txlpbc_val / txlpbc_test names
(register_txlpbc_hituav.py) unchanged - val/test are full, unmodified data
under both policies, so there is no separate "exact" val/test to register.
"""
import glob
import json
import os
import re

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

_ROOT = os.path.dirname(os.path.abspath(__file__))  # CDFSOD-benchmark/datasets
TXLPBC_ROOT = os.path.join(_ROOT, "TXL-PBC")
EXACT_DIR = os.path.join(TXLPBC_ROOT, "annotations_exact_instance")

_FNAME_RE = re.compile(r"train_(\d+)shot_seed(\d+)\.json$")


def _register(name, json_file, image_root):
    if name in DatasetCatalog.list():
        return
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    classes = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
    register_coco_instances(name, {}, json_file, image_root)
    MetadataCatalog.get(name).set(thing_classes=classes)


def register_txlpbc_exact_instance():
    if not os.path.isdir(EXACT_DIR):
        return
    image_root = os.path.join(TXLPBC_ROOT, "images", "train")
    for json_path in sorted(glob.glob(os.path.join(EXACT_DIR, "train_*shot_seed*.json"))):
        m = _FNAME_RE.search(os.path.basename(json_path))
        if not m:
            continue
        K, seed = m.group(1), m.group(2)
        name = f"txlpbc_exact_train_{K}shot_seed{seed}"
        _register(name, json_path, image_root)


register_txlpbc_exact_instance()
