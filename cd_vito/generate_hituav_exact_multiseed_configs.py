"""Generate CD-ViTO HIT-UAV exact-K-instance configs for all seeds (1..10),
using the new uniform hituav_exact_train_{K}shot_seed{N} registration
(datasets/register_hituav_exact_instance.py) instead of the old
hituav_train_{K}shot_seed1 (seed1-only) name.

Templates off the existing configs/hituav/vitl_shot{K}_hituav_finetune.yaml,
changing only:
  - DE.CLASS_PROTOTYPES -> prototypes_init/hituav_exact_train_{K}shot_seed{N}...
  - DATASETS.TRAIN -> ("hituav_exact_train_{K}shot_seed{N}",)
DATASETS.TEST stays ("hituav_test",) - unchanged, full test set reused as-is.
SOLVER/INPUT/TEST.EVAL_PERIOD/TOPK (shot-count-dependent, not seed-dependent)
are left untouched.

Output goes to a NEW directory, configs/hituav_exact_instance/, so
configs/hituav/*.yaml (and its prior seed1 results) are never modified.
"""
import argparse
import os

TEMPLATE_DIR = "configs/hituav"
OUT_DIR = "configs/hituav_exact_instance"
SHOTS = (1, 5, 10)


def generate(K, seed):
    template_path = os.path.join(TEMPLATE_DIR, f"vitl_shot{K}_hituav_finetune.yaml")
    with open(template_path, encoding="utf-8", newline="") as f:
        lines = f.readlines()

    old_proto = f"prototypes_init/hituav_train_{K}shot_seed1.vitl14.bbox.p{K}.sk.pkl"
    new_proto = f"prototypes_init/hituav_exact_train_{K}shot_seed{seed}.vitl14.bbox.p{K}.sk.pkl"
    old_train_line = f'  TRAIN: ("hituav_train_{K}shot_seed1",)\n'
    new_train_line = f'  TRAIN: ("hituav_exact_train_{K}shot_seed{seed}",)\n'

    n_proto = n_train = 0
    out_lines = []
    for line in lines:
        if line.strip() == f"CLASS_PROTOTYPES: {old_proto}":
            indent = line[:len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}CLASS_PROTOTYPES: {new_proto}\n")
            n_proto += 1
        elif line == old_train_line:
            out_lines.append(new_train_line)
            n_train += 1
        else:
            out_lines.append(line)

    if (n_proto, n_train) != (1, 1):
        raise RuntimeError(f"{template_path}: expected 1 substitution each for proto/train, got {(n_proto, n_train)}")
    assert '  TEST: ("hituav_test",)\n' in out_lines

    out_path = os.path.join(OUT_DIR, f"vitl_shot{K}_hituav_finetune_seed{seed}.yaml")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(out_lines)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in args.seeds:
        for K in args.shots:
            generate(K, seed)


if __name__ == "__main__":
    main()
