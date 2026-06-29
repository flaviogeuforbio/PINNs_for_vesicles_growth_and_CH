"""
run_multiseed.py
======================================================================
Run train_inverse_ac1d.py across several random seeds and aggregate the
per-run metrics into mean +/- std.

Each seed re-draws every source of randomness in the training script:
network initialization, the observed subset (data_fraction) and the
noise realization (data_noise). The aggregate therefore reflects the
full run-to-run variability, not noise alone.

The experiment (data_fraction, data_noise, ...) is fixed across seeds;
only --seed changes. Per-run metrics are read from
    artifacts/<run_name>_seed<seed>/metrics.json
and summarized into
    artifacts/<run_name>_multiseed.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # data / output
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # experiment definition (passed through to the training script)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--data_noise", type=float, default=0.0)

    # optional passthrough overrides (otherwise training-script defaults)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--pretrain_epochs", type=int, default=1000)
    parser.add_argument("--m_phi_init", type=float, default=0.2)

    parser.add_argument("--train_script", type=str, default="train_inverse_ac1d.py")

    return parser.parse_args()


# Metrics read back from each run's metrics.json and aggregated.
METRIC_KEYS = [
    "m_phi_final",
    "m_phi_relative_error",
    "full_grid_mse_phi",
    "full_grid_relative_l2_phi",
]


# ---------------------------------------------------------------------
# 2. Single-seed run
# ---------------------------------------------------------------------

def run_one_seed(args, seed):
    """Launch the training script for one seed and return its metrics dict."""
    run_name = f"{args.run_name}_seed{seed}"

    cmd = [
        sys.executable, args.train_script,
        "--data_path", args.data_path,
        "--run_name", run_name,
        "--seed", str(seed),
        "--data_fraction", str(args.data_fraction),
        "--data_noise", str(args.data_noise),
        "--epochs", str(args.epochs),
        "--pretrain_epochs", str(args.pretrain_epochs),
        "--m_phi_init", str(args.m_phi_init),
    ]

    print("=" * 80)
    print(f"SEED {seed}  ({run_name})")
    print("=" * 80)
    subprocess.run(cmd, check=True)

    metrics_path = Path("artifacts") / run_name / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# 3. Aggregation
# ---------------------------------------------------------------------

def aggregate(records, keys):
    """Mean and sample std (ddof=1) over seeds for each metric key."""
    summary = {}
    for key in keys:
        values = np.array([rec[key] for rec in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    records = []
    for seed in args.seeds:
        records.append(run_one_seed(args, seed))

    summary = aggregate(records, METRIC_KEYS)

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY")
    print("=" * 80)
    print(f"run_name      : {args.run_name}")
    print(f"seeds         : {args.seeds}")
    print(f"data_fraction : {args.data_fraction}")
    print(f"data_noise    : {args.data_noise}")
    print("-" * 80)
    for key in METRIC_KEYS:
        s = summary[key]
        print(f"{key:28s} : {s['mean']:.6e} +/- {s['std']:.2e}")
    print("=" * 80)

    out_path = Path("artifacts") / f"{args.run_name}_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_name": args.run_name,
                "seeds": args.seeds,
                "data_fraction": args.data_fraction,
                "data_noise": args.data_noise,
                "summary": summary,
            },
            f,
            indent=4,
        )
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
