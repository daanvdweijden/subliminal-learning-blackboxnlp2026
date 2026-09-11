#!/usr/bin/env python3
"""
Check whether the number-list carrier leaks the bias system prompt.

Compares raw_dataset.jsonl generated WITH a preference-bias system prompt
(e.g. "You love owls...") against the matching control dataset (no system
prompt), to see whether the generated number sequences are statistically
distinguishable, or contain literal mentions of the trait. This is a
diagnostic on whether the carrier itself leaks, not a measurement of
transmission in a finetuned model.

All the actual logic lives in notebooks/carrier_leakage_lib.py (also usable
directly from notebooks/carrier_leakage_eda.ipynb); this is a thin CLI.

Usage (single trait vs control):
    python scripts/check_carrier_leakage.py \\
        --bias-dataset data/runs/qwen2.5-7b/numbers/animal/owl/dataset/raw_dataset.jsonl \\
        --control-dataset data/runs/qwen2.5-7b/numbers/animal/control/dataset/raw_dataset.jsonl \\
        --trait-word owl

Usage (sweep every trait in a category against its control):
    python scripts/check_carrier_leakage.py \\
        --sweep --model qwen2.5-7b --task numbers --category animal \\
        [--csv-out results.csv]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
import carrier_leakage_lib as cl  # noqa: E402
from loguru import logger  # noqa: E402


def print_single_report(result: dict) -> None:
    print("\n=== Carrier leakage report ===")
    print(f"bias dataset:    {result['bias_dataset']} (n={result['n_bias']})")
    print(f"control dataset: {result['control_dataset']} (n={result['n_control']})")
    print(
        f"\nLiteral '{result['trait_word']}' mention rate: "
        f"bias={result['literal_leak_rate_bias']:.4%}  "
        f"control={result['literal_leak_rate_control']:.4%}"
    )
    print(f"\nNumber-distribution TVD (bias vs control): {result['tvd']:.4f}")
    print(f"Noise floor (control split-half TVD): {result['tvd_noise_floor']:.4f}")
    print(f"Chi-square (100 bins): chi2={result['chi2']:.1f}  p={result['chi2_p_value']:.4g}")
    print(f"Classifier AUC (bias vs control, 5-fold CV): {result['auc']:.4f}")
    print(f"\n  -> {'LIKELY LEAKS' if result['likely_leaks'] else 'no strong evidence of leakage'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bias-dataset")
    parser.add_argument("--control-dataset")
    parser.add_argument("--trait-word")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sweep", action="store_true", help="Sweep every trait in a category")
    parser.add_argument("--model", help="e.g. qwen2.5-7b (required with --sweep)")
    parser.add_argument("--task", default="numbers")
    parser.add_argument("--category", help="e.g. animal (required with --sweep)")
    parser.add_argument("--traits", nargs="*", help="Restrict sweep to these traits (default: all found locally)")
    parser.add_argument("--csv-out", help="Save sweep results to this CSV path")

    args = parser.parse_args()

    if args.sweep:
        if not args.model or not args.category:
            parser.error("--sweep requires --model and --category")
        df = cl.sweep_category(
            model=args.model,
            task=args.task,
            category=args.category,
            traits=args.traits,
            seed=args.seed,
        )
        if df.empty:
            logger.error("No traits analyzed — check that datasets are pulled locally.")
            sys.exit(1)
        cols = [
            "trait_word", "n_bias", "n_control",
            "literal_leak_rate_bias", "tvd", "tvd_noise_floor", "tvd_ratio",
            "chi2_p_value", "auc", "likely_leaks",
        ]
        print("\n=== Carrier leakage sweep:", f"{args.model}/{args.task}/{args.category} ===")
        print(df[cols].sort_values("auc", ascending=False).to_string(index=False))
        if args.csv_out:
            df[cols].to_csv(args.csv_out, index=False)
            logger.success(f"Saved to {args.csv_out}")
        return

    if not (args.bias_dataset and args.control_dataset and args.trait_word):
        parser.error("Non-sweep mode requires --bias-dataset, --control-dataset, and --trait-word")

    for p in (args.bias_dataset, args.control_dataset):
        if not Path(p).exists():
            logger.error(f"Missing file: {p} (expected under data/runs/<model>/<task>/<category>/<trait>/dataset/)")
            sys.exit(1)

    result = cl.analyze_pair(
        args.bias_dataset,
        args.control_dataset,
        args.trait_word,
        task=args.task,
        seed=args.seed,
    )
    print_single_report(result)


if __name__ == "__main__":
    main()
