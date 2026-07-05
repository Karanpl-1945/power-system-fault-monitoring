from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a rare-fault evaluation set by resampling existing test windows."
    )
    parser.add_argument("--input", default="data/processed/features_test.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fault-ratio", type=float, required=True)
    parser.add_argument("--max-faults", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fault_ratio < 1:
        raise ValueError("--fault-ratio must be between 0 and 1")

    data = pd.read_csv(args.input)
    fault = data[data["binary_target"] == 1]
    normal = data[data["binary_target"] == 0]

    n_fault = len(fault) if args.max_faults is None else min(args.max_faults, len(fault))
    n_normal_needed = int(round(n_fault * (1 - args.fault_ratio) / args.fault_ratio))

    if n_normal_needed > len(normal):
        n_normal_needed = len(normal)
        n_fault = int(round(n_normal_needed * args.fault_ratio / (1 - args.fault_ratio)))

    sampled_fault = fault.sample(n=n_fault, random_state=args.seed)
    sampled_normal = normal.sample(n=n_normal_needed, random_state=args.seed)
    realistic = (
        pd.concat([sampled_fault, sampled_normal], axis=0)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    output = args.output
    if output is None:
        pct = int(round(args.fault_ratio * 100))
        output = f"data/processed/features_test_realistic_{pct}pct.csv"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    realistic.to_csv(output_path, index=False)

    actual_fault_ratio = float(realistic["binary_target"].mean())
    print(f"input={args.input}")
    print(f"output={output_path}")
    print(f"rows={len(realistic)}")
    print(f"fault_rows={int(realistic['binary_target'].sum())}")
    print(f"normal_rows={int((realistic['binary_target'] == 0).sum())}")
    print(f"fault_ratio={actual_fault_ratio:.4f}")


if __name__ == "__main__":
    main()
