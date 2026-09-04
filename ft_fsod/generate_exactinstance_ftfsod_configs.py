"""Generate FT-FSOD (mmdet) TXL-PBC configs for the exact-K-instance few-shot
policy (Dataset/TXL-PBC_Dataset/TXL-PBC/annotations_exact_instance/).

Templates off the existing K-image policy config
configs_cdfsod/final_configs_preliminary_valsel/grounding_dino_swin-b_finetune_TXL-PBC_{K}shot_seed1.py
and only rewrites the single `ann_file=...` line in train_dataloader to point
at annotations_exact_instance/train_{K}shot_seed{seed}.json. Everything else
(model, schedule, val/test which stay on the full annotations_std/val.json
and test.json) is left untouched, since the K-image configs for 1/5/10-shot
only ever differed in that one line (verified by diff).

Output goes to configs_cdfsod/final_configs_exact_instance/ - a directory
that does not exist under the K-image policy, so nothing there is ever
overwritten by this script.

Usage:
    python generate_exactinstance_ftfsod_configs.py [--seeds 1] [--shots 1 5 10]
"""
import argparse
import os
import re

TEMPLATE_DIR = "configs_cdfsod/final_configs_preliminary_valsel"
OUT_DIR = "configs_cdfsod/final_configs_exact_instance"
DATASET = "TXL-PBC"

OLD_ANN_RE = re.compile(r"ann_file=f'annotations_std/train_\d+shot_seed1\.json'")


def generate(K, seed):
    template_path = os.path.join(TEMPLATE_DIR, f"grounding_dino_swin-b_finetune_{DATASET}_{K}shot_seed1.py")
    with open(template_path, encoding="utf-8", newline="") as f:
        text = f.read()

    new_ann_line = f"ann_file=f'annotations_exact_instance/train_{K}shot_seed{seed}.json'"
    new_text, n_sub = OLD_ANN_RE.subn(new_ann_line, text)
    if n_sub != 1:
        raise RuntimeError(f"{template_path}: expected exactly 1 ann_file substitution, got {n_sub}")

    out_path = os.path.join(OUT_DIR, f"grounding_dino_swin-b_finetune_{DATASET}_{K}shot_seed{seed}.py")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    print(f"wrote {out_path} (ann_file -> annotations_exact_instance/train_{K}shot_seed{seed}.json)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in args.seeds:
        for K in args.shots:
            generate(K, seed)


if __name__ == "__main__":
    main()
