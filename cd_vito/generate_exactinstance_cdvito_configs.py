"""Generate CD-ViTO (Detectron2) TXL-PBC configs for the exact-K-instance
few-shot policy, mirroring FT-FSOD's counterpart
(../FT-FSOD/generate_exactinstance_ftfsod_configs.py).

Templates off the existing K-image policy config
configs/txlpbc/vitl_shot{K}_txlpbc_finetune.yaml and only rewrites:
  - DE.CLASS_PROTOTYPES: points at the new per-seed prototype file
    (prototypes_init/txlpbc_exact_train_{K}shot_seed{N}.vitl14.bbox.p{K}.sk.pkl)
  - DATASETS.TRAIN: ("txlpbc_exact_train_{K}shot_seed{N}",)
DATASETS.TEST stays ("txlpbc_test",) - full, unmodified test set, same as the
K-image configs. SOLVER/INPUT/TEST.EVAL_PERIOD (which scale with shot count,
not with the sampling policy) are left untouched.

Output goes to configs/txlpbc_exact_instance/ - does not exist under the
K-image policy, so nothing is ever overwritten.

Usage:
    python generate_exactinstance_cdvito_configs.py [--seeds 1] [--shots 1 5 10]
"""
import argparse
import os

TEMPLATE_DIR = "configs/txlpbc"
OUT_DIR = "configs/txlpbc_exact_instance"


def generate(K, seed):
    template_path = os.path.join(TEMPLATE_DIR, f"vitl_shot{K}_txlpbc_finetune.yaml")
    with open(template_path, encoding="utf-8", newline="") as f:
        lines = f.readlines()

    old_proto = f"prototypes_init/txlpbc_train_{K}shot_seed1.vitl14.bbox.p{K}.sk.pkl"
    new_proto = f"prototypes_init/txlpbc_exact_train_{K}shot_seed{seed}.vitl14.bbox.p{K}.sk.pkl"
    old_train = f'DATASETS:\n'  # marker only, real substitution below by exact-line match
    old_train_line = f'  TRAIN: ("txlpbc_train_{K}shot_seed1",)\n'
    new_train_line = f'  TRAIN: ("txlpbc_exact_train_{K}shot_seed{seed}",)\n'

    n_proto_sub = 0
    n_train_sub = 0
    out_lines = []
    for line in lines:
        if line.strip() == f"CLASS_PROTOTYPES: {old_proto}":
            indent = line[:len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}CLASS_PROTOTYPES: {new_proto}\n")
            n_proto_sub += 1
        elif line == old_train_line:
            out_lines.append(new_train_line)
            n_train_sub += 1
        else:
            out_lines.append(line)

    if n_proto_sub != 1:
        raise RuntimeError(f"{template_path}: expected 1 CLASS_PROTOTYPES substitution, got {n_proto_sub}")
    if n_train_sub != 1:
        raise RuntimeError(f"{template_path}: expected 1 DATASETS.TRAIN substitution, got {n_train_sub}")

    out_path = os.path.join(OUT_DIR, f"vitl_shot{K}_txlpbc_finetune_seed{seed}.yaml")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(out_lines)
    print(f"wrote {out_path} (TRAIN -> txlpbc_exact_train_{K}shot_seed{seed}, "
          f"CLASS_PROTOTYPES -> {new_proto})")


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
