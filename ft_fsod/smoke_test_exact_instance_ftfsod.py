"""Smoke test: confirm the mmdet CocoDataset actually loads exactly 3/15/30
annotations (1 per image) from the exact-K-instance TXL-PBC configs under
configs_cdfsod/final_configs_exact_instance/.

Usage: python smoke_test_exact_instance_ftfsod.py [--seeds 1] [--shots 1 5 10]
"""
import argparse

from mmengine import Config
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import DATASETS

CONFIG_DIR = "configs_cdfsod/final_configs_exact_instance"
DATASET = "TXL-PBC"


def check(K, seed):
    cfg = Config.fromfile(f"{CONFIG_DIR}/grounding_dino_swin-b_finetune_{DATASET}_{K}shot_seed{seed}.py")
    ds_cfg = cfg.train_dataloader["dataset"]
    ds = DATASETS.build(ds_cfg)
    n_img = len(ds)
    per_img = [len(ds.get_data_info(i)["instances"]) for i in range(n_img)]
    n_ann = sum(per_img)
    ok = (n_img == 3 * K and n_ann == 3 * K and all(c == 1 for c in per_img))
    print(f"seed{seed}/{K}shot: images={n_img} annotations={n_ann} "
          f"ann_file={ds_cfg['ann_file']} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    all_ok = True
    for seed in args.seeds:
        for K in args.shots:
            all_ok &= check(K, seed)

    print("ALL_OK" if all_ok else "SOME_FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
