

"""
plot_results.py
Two modes:
  1) scatter : from a single per-direction association TSV
  2) top10   : from the merged TSV (after merge_iqr.py), show top-N by direction

Inputs must be TSVs produced by the companion scripts so columns exist:
  scatter: needs Raw_X_Normal, Raw_X_Condition, Association_Score_Scaled (or AS_Banded_Scaled)
  top10  : needs Direction and AS_Banded_Scaled
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_scatter(tsv, out_png, title=None, y_col=None):
    df = pd.read_csv(tsv, sep="\t")
    # pick Y column
    y = y_col if y_col in df.columns else ("Association_Score_Scaled" if "Association_Score_Scaled" in df.columns else "AS_Banded_Scaled")
    if y not in df.columns:
        raise ValueError("Cannot find a Y column (Association_Score_Scaled or AS_Banded_Scaled).")

    plt.figure(figsize=(12, 7))
    plt.scatter(df["Raw_X_Normal"], df[y], s=30, label="Normal raw score", alpha=0.9)
    plt.scatter(df["Raw_X_Condition"], df[y], s=30, label="Condition raw score", alpha=0.9)

    plt.title(title or "Association Score vs Raw Scores", fontsize=14)
    plt.xlabel("Original Scores (reliability)", fontsize=12)
    plt.ylabel(f"{y}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_top10(merged_tsv, out_png, top_n=10):
    df = pd.read_csv(merged_tsv, sep="\t")
    y = "AS_Banded_Scaled" if "AS_Banded_Scaled" in df.columns else "Association_Score_Scaled"
    if "Direction" not in df.columns:
        raise ValueError("Merged TSV must contain 'Direction' column.")

    # Split by direction
    a = df[df["Direction"] == "N->C"].copy()
    b = df[df["Direction"] == "C->N"].copy()

    a_top = a.sort_values(y, ascending=False).head(top_n)
    b_top = b.sort_values(y, ascending=False).head(top_n)

    # Build a tall figure with two tables and small histograms for distribution context
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1, 1], hspace=0.25, wspace=0.25)

    # Panel A: Table for N->C
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.set_title(f"Top {top_n}: Direction N→C (AS={y})", fontsize=12)
    tbl_a = a_top[["Gene_ID", "Gene_Name", "Association_Score", y]].copy()
    ax_a.table(cellText=tbl_a.values, colLabels=tbl_a.columns, loc="center", cellLoc="left", colLoc="left")

    # Panel A hist
    ax_ah = fig.add_subplot(gs[0, 1])
    ax_ah.hist(a[y].dropna().values, bins=30)
    ax_ah.set_title("N→C distribution")
    ax_ah.set_xlabel(y)
    ax_ah.set_ylabel("Count")

    # Panel B: Table for C->N
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.axis("off")
    ax_b.set_title(f"Top {top_n}: Direction C→N (AS={y})", fontsize=12)
    tbl_b = b_top[["Gene_ID", "Gene_Name", "Association_Score", y]].copy()
    ax_b.table(cellText=tbl_b.values, colLabels=tbl_b.columns, loc="center", cellLoc="left", colLoc="left")

    # Panel B hist
    ax_bh = fig.add_subplot(gs[1, 1])
    ax_bh.hist(b[y].dropna().values, bins=30)
    ax_bh.set_title("C→N distribution")
    ax_bh.set_xlabel(y)
    ax_bh.set_ylabel("Count")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plotting utilities for association-score outputs.")
    ap.add_argument("--mode", required=True, choices=["scatter", "top10"], help="Choose plotting mode.")
    ap.add_argument("--in", dest="inp", required=True, help="Input TSV (single for scatter; merged for top10).")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--title", help="Title for scatter.")
    ap.add_argument("--y-col", help="Explicit Y column (Association_Score_Scaled or AS_Banded_Scaled).")
    ap.add_argument("--top-n", type=int, default=10, help="Top-N for summary (only for top10 mode).")
    args = ap.parse_args()

    if args.mode == "scatter":
        plot_scatter(args.inp, args.out, title=args.title, y_col=args.y_col)
    else:
        plot_top10(args.inp, args.out, top_n=args.top_n)


if __name__ == "__main__":
    main()
