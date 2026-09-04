"""Generate HIT-UAV exact-K-instance FT-FSOD configs for seed2..10.

The existing configs_cdfsod/final_configs_preliminary_valsel/
grounding_dino_swin-b_finetune_HIT-UAV_{K}shot_seed1.py already has a clean
val(!=test) protocol (val_dataloader -> normal_json/annotations_std/val.json,
test_dataloader -> normal_json/annotations_std/test.json) and trains on the
correct exact-instance seed1 support set - no fixing needed, only the
train ann_file's seed1 -> seedN needs to change to pick up the new
Dataset/HIT-UAV-Infrared-Thermal-Dataset/fewshot/train_{K}shot_seed{N}.json
files (seed2..10, generated separately).

Output goes to a NEW directory, final_configs_hituav_exact_multiseed/, so
final_configs_preliminary_valsel/ and its exp_preliminary_results/ are never
touched.
"""
import argparse
import os

TEMPLATE_DIR = "configs_cdfsod/final_configs_preliminary_valsel"
OUT_DIR = "configs_cdfsod/final_configs_hituav_exact_multiseed"
SHOTS = [1, 5, 10]


def generate(K, seed):
    template_path = os.path.join(TEMPLATE_DIR, f"grounding_dino_swin-b_finetune_HIT-UAV_{K}shot_seed1.py")
    with open(template_path, encoding="utf-8", newline="") as f:
        text = f.read()

    old = f"ann_file=f'fewshot/train_{K}shot_seed1.json'"
    new = f"ann_file=f'fewshot/train_{K}shot_seed{seed}.json'"
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{template_path}: expected 1 occurrence of {old!r}, got {n}")
    text = text.replace(old, new, 1)

    assert "ann_file=f'normal_json/annotations_std/val.json'" in text
    assert "ann_file=f'normal_json/annotations_std/test.json'" in text

    out_path = os.path.join(OUT_DIR, f"grounding_dino_swin-b_finetune_HIT-UAV_{K}shot_seed{seed}.py")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in args.seeds:
        for K in args.shots:
            generate(K, seed)


if __name__ == "__main__":
    main()
