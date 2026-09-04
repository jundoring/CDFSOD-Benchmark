"""Generate fair-conditions CD-ViTO SIXray-D exact-K-instance seed1 configs.

Templates off the existing configs/sixray/vitl_shot{K}_sixray_finetune.yaml,
changing only:
  - DE.CLASS_PROTOTYPES -> new per-shot prototype file built from the
    canonical exact-instance seed1 support set
  - DATASETS.TRAIN -> ("sixray_exact_train_{K}shot_seed1",)
  - DATASETS.TEST  -> ("sixray_exact_test",)   (was "SIXRAY_test", which
    resolves to the non-canonical CDFSOD-benchmark/datasets/SIXRAY copy -
    replaced with the canonical Dataset/SIXray-D/SIXray-D test.json)

SOLVER/INPUT/TEST.EVAL_PERIOD (shot-count-dependent, not data-source-
dependent) are left untouched. Output goes to a new directory,
configs/sixray_exact_instance_seed1_clean/, so the existing
configs/sixray/*.yaml are never modified.
"""
import argparse
import os

TEMPLATE_DIR = "configs/sixray"
OUT_DIR = "configs/sixray_exact_instance_seed1_clean"
SHOTS = (1, 5, 10)


def generate(K, seed):
    template_path = os.path.join(TEMPLATE_DIR, f"vitl_shot{K}_sixray_finetune.yaml")
    with open(template_path, encoding="utf-8", newline="") as f:
        lines = f.readlines()

    old_proto = f"prototypes_init/SIXRAY_{K}shot.vitl14.bbox.p{K}.sk.pkl"
    new_proto = f"prototypes_init/sixray_exact_train_{K}shot_seed{seed}.vitl14.bbox.p{K}.sk.pkl"
    old_train_line = f'  TRAIN: ("SIXRAY_{K}shot",)\n'
    new_train_line = f'  TRAIN: ("sixray_exact_train_{K}shot_seed{seed}",)\n'
    old_test_line = '  TEST: ("SIXRAY_test",)\n'
    new_test_line = '  TEST: ("sixray_exact_test",)\n'

    n_proto = n_train = n_test = 0
    out_lines = []
    for line in lines:
        if line.strip() == f"CLASS_PROTOTYPES: {old_proto}":
            indent = line[:len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}CLASS_PROTOTYPES: {new_proto}\n")
            n_proto += 1
        elif line == old_train_line:
            out_lines.append(new_train_line)
            n_train += 1
        elif line == old_test_line:
            out_lines.append(new_test_line)
            n_test += 1
        else:
            out_lines.append(line)

    if (n_proto, n_train, n_test) != (1, 1, 1):
        raise RuntimeError(
            f"{template_path}: expected 1 substitution each for proto/train/test, "
            f"got {(n_proto, n_train, n_test)}"
        )

    out_path = os.path.join(OUT_DIR, f"vitl_shot{K}_sixray_finetune_seed{seed}.yaml")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(out_lines)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in args.seeds:
        for K in args.shots:
            generate(K, seed)


if __name__ == "__main__":
    main()
