#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
association_score.py
Compute Association Scores between a Normal state and a Condition (disease or treatment).

Direction A (N->C off): AS = ST_N + SD_C
Direction B (C->N off): AS = ST_C + SD_N

Inputs are four TSV files (3 columns each): Gene_ID, Gene_Name, Score
  --normal-transcribed   -> ST_N
  --normal-dormant       -> SD_N
  --cond-transcribed     -> ST_C
  --cond-dormant         -> SD_C

Output TSV includes:
  Gene_ID, Gene_Name, ST_N, SD_N, ST_C, SD_C, Association_Score, Association_Score_Scaled,
  Direction, Raw_X_Normal, Raw_X_Condition

Optional: save a scatter plot with both raw X series vs. scaled AS (blue=normal, red=condition).
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _read_score_tsv(p: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(p, sep="\t", header=None, names=["Gene_ID", "Gene_Name", colname],
                     dtype={"Gene_ID": str, "Gene_Name": str})
    # Score to numeric, coerce + abs
    df[colname] = pd.to_numeric(df[colname], errors="coerce").fillna(0).abs()
    # Ensure unique rows on (Gene_ID,Gene_Name); if dup, keep max score
    df = df.groupby(["Gene_ID", "Gene_Name"], as_index=False)[colname].max()
    return df


def _minmax_to_1_100(x: pd.Series) -> pd.Series:
    xmin, xmax = float(x.min()), float(x.max())
    if xmax == xmin:
        # All equal -> map to 50
        return pd.Series(np.full_like(x, 50.0, dtype=float), index=x.index)
    return 1.0 + (x - xmin) * (99.0 / (xmax - xmin))


def compute_association(nt, nd, ct, cd, direction: str) -> pd.DataFrame:
    # Load each file
    st_n = _read_score_tsv(nt, "ST_N")
    sd_n = _read_score_tsv(nd, "SD_N")
    st_c = _read_score_tsv(ct, "ST_C")
    sd_c = _read_score_tsv(cd, "SD_C")

    # Outer merge to form union of genes
    df = st_n.merge(sd_n, on=["Gene_ID", "Gene_Name"], how="outer")
    df = df.merge(st_c, on=["Gene_ID", "Gene_Name"], how="outer")
    df = df.merge(sd_c, on=["Gene_ID", "Gene_Name"], how="outer")
    df[["ST_N", "SD_N", "ST_C", "SD_C"]] = df[["ST_N", "SD_N", "ST_C", "SD_C"]].fillna(0).astype(float)

    # Compute AS by direction
    direction = direction.upper()
    if direction not in {"A", "B"}:
        raise ValueError("direction must be 'A' or 'B'")

    if direction == "A":
        # Normal active, Condition off
        df["Association_Score"] = df["ST_N"] + df["SD_C"]
        df["Direction"] = "N->C"
        # Raw X series for plotting per your convention
        df["Raw_X_Normal"] = df["ST_N"]
        df["Raw_X_Condition"] = df["SD_C"]
    else:
        # Condition active, Normal off
        df["Association_Score"] = df["ST_C"] + df["SD_N"]
        df["Direction"] = "C->N"
        df["Raw_X_Normal"] = df["SD_N"]
        df["Raw_X_Condition"] = df["ST_C"]

    # Scale to [1,100]
    df["Association_Score_Scaled"] = _minmax_to_1_100(df["Association_Score"]).round(2)

    # Sort by scaled descending
    df = df.sort_values("Association_Score_Scaled", ascending=False).reset_index(drop=True)
    return df


def save_plot(df: pd.DataFrame, title: str, out_png: str):
    plt.figure(figsize=(12, 7))
    # Blue = "normal side" raw score, Red = "condition side" raw score
    plt.scatter(df["Raw_X_Normal"], df["Association_Score_Scaled"], s=30, label="Normal raw score", alpha=0.9)
    plt.scatter(df["Raw_X_Condition"], df["Association_Score_Scaled"], s=30, label="Condition raw score", alpha=0.9)

    plt.title(title, fontsize=14)
    plt.xlabel("Original Scores (reliability)", fontsize=12)
    plt.ylabel("Association Score (Scaled 1–100)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compute Association Scores (Direction A or B).")
    ap.add_argument("--normal-transcribed", required=True, help="TSV with 3 cols -> ST_N")
    ap.add_argument("--normal-dormant",   required=True, help="TSV with 3 cols -> SD_N")
    ap.add_argument("--cond-transcribed", required=True, help="TSV with 3 cols -> ST_C")
    ap.add_argument("--cond-dormant",     required=True, help="TSV with 3 cols -> SD_C")
    ap.add_argument("--direction", choices=["A", "B"], required=True,
                    help="A: ST_N + SD_C   |   B: ST_C + SD_N")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--plot", help="Optional: output PNG path for scatter")
    ap.add_argument("--title", help="Optional: plot title")
    args = ap.parse_args()

    df = compute_association(
        nt=args.normal_transcribed,
        nd=args.normal_dormant,
        ct=args.cond_transcribed,
        cd=args.cond_dormant,
        direction=args.direction
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)

    if args.plot:
        title = args.title or f"Association Score (dir {args.direction})"
        save_plot(df, title=title, out_png=args.plot)


if __name__ == "__main__":
    main()
