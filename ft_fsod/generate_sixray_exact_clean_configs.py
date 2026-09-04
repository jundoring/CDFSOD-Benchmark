"""Generate a fair-conditions SIXray-D exact-K-instance seed1 FT-FSOD config.

The existing configs_cdfsod/final_configs_sixray/grounding_dino_swin-b_finetune_SIXray-D_{K}shot.py
already train on the correct exact-instance support set
(fewshot/train_{K}shot_seed1.json), but val_dataloader/val_evaluator both
point at test.json - meaning checkpoint selection (save_best='coco/bbox_mAP')
happens against the test set itself, every epoch, for the full 100-epoch
schedule. That is not a fair/clean protocol.

This script only changes val_dataloader's and val_evaluator's ann_file from
test.json to val.json (the dataset's real held-out validation split,
Dataset/SIXray-D/SIXray-D/val.json - 1005 images, disjoint from test.json's
836 images). train_dataloader's ann_file (fewshot/train_{K}shot_seed1.json)
and test_dataloader/test_evaluator's ann_file (test.json, evaluated once
after training, via the best-on-val checkpoint) are left untouched.

Output goes to a new directory, final_configs_sixray_exact_seed1_clean/, so
the existing final_configs_sixray/ configs and their results
(exp_sixray_results/) are never modified or overwritten.
"""
import argparse
import os
import re

TEMPLATE_DIR = "configs_cdfsod/final_configs_sixray"
OUT_DIR = "configs_cdfsod/final_configs_sixray_exact_seed1_clean"
SHOTS = [1, 5, 10]

VAL_ANN_OLD = "        ann_file=f'test.json',\n        data_prefix=dict(img='images/'),\n        pipeline=test_pipeline,\n        return_classes=True))\n\ntest_dataloader"
VAL_ANN_NEW = "        ann_file=f'val.json',\n        data_prefix=dict(img='images/'),\n        pipeline=test_pipeline,\n        return_classes=True))\n\ntest_dataloader"

VAL_EVAL_OLD = "val_evaluator = dict(\n    ann_file=f'{data_root}/test.json',"
VAL_EVAL_NEW = "val_evaluator = dict(\n    ann_file=f'{data_root}/val.json',"


def generate(K, seed):
    """Template is always seed1-flavored on disk; for seed!=1 we additionally
    swap the train ann_file's seed1 -> seedN (the template's train/test split
    and schedule don't depend on the data seed, only which support file is
    loaded)."""
    template_path = os.path.join(TEMPLATE_DIR, f"grounding_dino_swin-b_finetune_SIXray-D_{K}shot.py")
    with open(template_path, encoding="utf-8") as f:
        text = f.read()

    if VAL_ANN_OLD not in text:
        raise RuntimeError(f"{template_path}: val_dataloader ann_file block not found as expected")
    text = text.replace(VAL_ANN_OLD, VAL_ANN_NEW, 1)

    if VAL_EVAL_OLD not in text:
        raise RuntimeError(f"{template_path}: val_evaluator ann_file block not found as expected")
    text = text.replace(VAL_EVAL_OLD, VAL_EVAL_NEW, 1)

    train_ann_old = f"ann_file=f'fewshot/train_{K}shot_seed1.json'"
    if seed != 1:
        train_ann_new = f"ann_file=f'fewshot/train_{K}shot_seed{seed}.json'"
        assert text.count(train_ann_old) == 1
        text = text.replace(train_ann_old, train_ann_new, 1)
        train_ann_old = train_ann_new

    # sanity: train ann_file must be the exact-instance file for this seed, and
    # test_dataloader/test_evaluator must still be test.json (untouched)
    assert train_ann_old in text
    assert text.count("ann_file=f'test.json'") == 1  # only test_dataloader now
    assert "ann_file=f'{data_root}/test.json'" in text  # test_evaluator untouched

    out_path = os.path.join(OUT_DIR, f"grounding_dino_swin-b_finetune_SIXray-D_{K}shot_seed{seed}.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in args.seeds:
        for K in args.shots:
            generate(K, seed)


if __name__ == "__main__":
    main()
