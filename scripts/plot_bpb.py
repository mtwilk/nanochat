"""
Plot training and validation bits-per-byte (bpb) curves over the course of
pretraining, from the metrics.csv written by scripts.base_train.

Usage:
python -m scripts.plot_bpb --depth=4
python -m scripts.plot_bpb --metrics-csv=/path/to/metrics.csv --out=bpb.png
"""
import os
import csv
import argparse

import matplotlib.pyplot as plt

from nanochat.common import get_base_dir

parser = argparse.ArgumentParser(description="Plot train/val bpb curves from base_train metrics.csv")
parser.add_argument("--depth", type=int, default=None, help="model depth, used to locate the default checkpoint dir (e.g. 4 -> d4)")
parser.add_argument("--model-tag", type=str, default=None, help="model tag, overrides --depth for locating the checkpoint dir")
parser.add_argument("--metrics-csv", type=str, default="", help="path to metrics.csv (overrides --depth/--model-tag lookup)")
parser.add_argument("--out", type=str, default="", help="output image path (default: <checkpoint_dir>/bpb_curve.png)")
args = parser.parse_args()

if args.metrics_csv:
    metrics_csv_path = args.metrics_csv
    out_path = args.out if args.out else os.path.join(os.path.dirname(metrics_csv_path), "bpb_curve.png")
else:
    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    metrics_csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    out_path = args.out if args.out else os.path.join(checkpoint_dir, "bpb_curve.png")

steps, train_bpb, val_bpb = [], [], []
with open(metrics_csv_path, newline="") as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"]))
        train_bpb.append(float(row["train_eval_bpb"]))
        val_bpb.append(float(row["val_bpb"]))

plt.figure(figsize=(8, 5))
plt.plot(steps, train_bpb, label="train bpb", alpha=0.7)
plt.plot(steps, val_bpb, label="val bpb", alpha=0.7)
plt.xlabel("step")
plt.ylabel("bits per byte")
plt.title("Pretraining bpb curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
