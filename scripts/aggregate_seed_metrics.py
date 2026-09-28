"""
Aggregate per-seed test_metrics JSON files (from src.evaluate --tag) into a
mean +/- std table.

CLI: python scripts/aggregate_seed_metrics.py results/v2/test_metrics_seed*.json
"""

import argparse
import json
import statistics as stats
from pathlib import Path

METRICS = ["accuracy", "auroc", "precision", "recall", "f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    runs = [json.loads(p.read_text()) for p in args.files]

    header = f"{'file':<40}" + "".join(f"{m:>10}" for m in METRICS)
    print(header)
    for p, r in zip(args.files, runs):
        print(f"{p.name:<40}" + "".join(f"{r[m]:>10.4f}" for m in METRICS))

    print("-" * len(header))
    means = {m: stats.mean(r[m] for r in runs) for m in METRICS}
    stdevs = {m: (stats.stdev(r[m] for r in runs) if len(runs) > 1 else 0.0) for m in METRICS}
    print(f"{'mean':<40}" + "".join(f"{means[m]:>10.4f}" for m in METRICS))
    print(f"{'std':<40}" + "".join(f"{stdevs[m]:>10.4f}" for m in METRICS))


if __name__ == "__main__":
    main()
