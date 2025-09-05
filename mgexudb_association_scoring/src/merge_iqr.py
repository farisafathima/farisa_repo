


"""
merge_iqr.py
Merge two Association-Score TSVs (Direction A and B), and optionally apply IQR-based
outlier handling with banded scaling.

Expected input columns (from association_score.py):
  Gene_ID, Gene_Name, ST_N, SD_N, ST_C, SD_C, Association_Score, Association_Score_Scaled,
  Direction, Raw_X_Normal, Raw_X_Condition

Outputs:
  merged TSV with a new 'AS_Banded_Scaled' column if --iqr true
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def _minmax(x: pd.Series, lo: float, hi: float) -> pd.Series:
    xmin, xmax = float(x.min()), float(x.max())
    if xmax == xmin:
        return pd.Series(np.full_like(x, (lo + hi) / 2.0, dtype=float), index=x.index)
    return lo + (x - xmin) * ((hi - lo) / (xmax - xmin))


def banded_scale_with_iqr(as_series: pd.Series,
                          bottom_band=(1.0, 2.0),
                          mid_band=(2.0, 99.0),
                          top_band=(99.0, 100.0)) -> pd.Series:
    # Identify outliers via IQR rule on raw AS
    q1, q3 = as_series.quantile(0.25), as_series.quantile(0.75)
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    lo = q1 - 1.5 * iqr

    mask_top = as_series > hi
    mask_bot = as_series < lo
    mask_mid = ~(mask_top | mask_bot)

    scaled = pd.Series(index=as_series.index, dtype=float)

    if mask_bot.any():
        scaled.loc[mask_bot] = _minmax(as_series[mask_bot], bottom_band[0], bottom_band[1])
    if mask_mid.any():
        scaled.loc[mask_mid] = _minmax(as_series[mask_mid], mid_band[0], mid_band[1])
    if mask_top.any():
        scaled.loc[mask_top] = _minmax(as_series[mask_top], top_band[0], top_band[1])

    return scaled.round(2)


def main():
    ap = argparse.ArgumentParser(description="Merge A+B association TSVs and optionally apply IQR-banded scaling.")
    ap.add_argument("--a", required=True, help="Association TSV for Direction A (N->C)")
    ap.add_argument("--b", required=True, help="Association TSV for Direction B (C->N)")
    ap.add_argument("--out", required=True, help="Output merged TSV")
    ap.add_argument("--iqr", default="true", choices=["true", "false"],
                    help="Apply IQR-banded scaling (true/false), default true")
    ap.add_argument("--bottom-band", nargs=2, type=float, default=[1.0, 2.0], metavar=("LOW", "HIGH"))
    ap.add_argument("--mid-band",    nargs=2, type=float, default=[2.0, 99.0], metavar=("LOW", "HIGH"))
    ap.add_argument("--top-band",    nargs=2, type=float, default=[99.0, 100.0], metavar=("LOW", "HIGH"))
    args = ap.parse_args()

    a = pd.read_csv(args.a, sep="\t")
    b = pd.read_csv(args.b, sep="\t")

    # Minimal column check
    for df in (a, b):
        if "Association_Score" not in df.columns or "Direction" not in df.columns:
            raise ValueError("Input TSVs must be produced by association_score.py")

    merged = pd.concat([a, b], ignore_index=True)

    if args.iqr == "true":
        merged["AS_Banded_Scaled"] = banded_scale_with_iqr(
            merged["Association_Score"],
            bottom_band=tuple(args.bottom_band),
            mid_band=tuple(args.mid_band),
            top_band=tuple(args.top_band),
        )
    else:
        # If not using IQR banding, just re-normalize merged set to [1,100]
        merged["AS_Banded_Scaled"] = 1.0 + (merged["Association_Score"] - merged["Association_Score"].min()) * \
                                     (99.0 / (merged["Association_Score"].max() - merged["Association_Score"].min()))
        merged["AS_Banded_Scaled"] = merged["AS_Banded_Scaled"].round(2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
