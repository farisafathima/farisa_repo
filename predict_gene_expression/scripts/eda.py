from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_CSV = Path(r"C:\Users\faris\SEM_III_PROJECT\PROJECT\de_train_enriched.csv")
OUT_DIR   = Path(r"C:\Users\faris\SEM_III_PROJECT\PROJECT\eda_out")
LAST_EMBED_COL = "embedding_767"
TOPK_CLUSTermap_GENES = 50
PREFERRED_GENE = "A1BG"


sns.set_context("talk")
OUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_CSV)
print(f"[input] {INPUT_CSV.name} shape={df.shape}")


if LAST_EMBED_COL not in df.columns:
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if not emb_cols:
        raise RuntimeError(f"No 'embedding_*' columns found and '{LAST_EMBED_COL}' missing.")
    def _suffix_num(c):
        try: return int(c.split("_", 1)[1])
        except: return -1
    LAST_EMBED_COL = sorted(emb_cols, key=_suffix_num)[-1]

start_idx = df.columns.get_loc(LAST_EMBED_COL) + 1
gene_cols = df.columns[start_idx:].tolist()
if not gene_cols:
    raise RuntimeError(f"No columns found after '{LAST_EMBED_COL}'.")

# Group column (prefer cell_type, else sm_name)
GROUP_COL = "cell_type" if "cell_type" in df.columns else "sm_name"
if GROUP_COL not in df.columns:
    raise KeyError("Neither 'cell_type' nor 'sm_name' found for grouping.")

# -------- Summary text (eda_summary.txt) --------
lines = []
lines.append(f"File: {INPUT_CSV}")
lines.append(f"Shape: {df.shape}")
lines.append(f"Last embedding column: {LAST_EMBED_COL} (index {start_idx-1})")
lines.append(f"Detected gene columns: {len(gene_cols)}")
lines.append(f"First 5 gene columns: {gene_cols[:5]}")
lines.append(f"Group column used: {GROUP_COL} (n={df[GROUP_COL].nunique(dropna=True)} levels)")
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False).head(20)
lines.append("\nTop missing columns (<=20):")
lines.extend([f"  {k}: {v}" for k, v in miss.items()])
(OUT_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")

#1) Violin: ONE gene by group 
gene_for_violin = PREFERRED_GENE if PREFERRED_GENE in gene_cols else gene_cols[0]
plt.figure(figsize=(12, 6))
sns.violinplot(data=df[[GROUP_COL, gene_for_violin]].dropna(),
               x=GROUP_COL, y=gene_for_violin, inner="quartile", cut=0)
plt.title(f"{gene_for_violin} expression by {GROUP_COL} (violin)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / f"1_violin_{gene_for_violin}_by_{GROUP_COL}.png", dpi=150)
plt.close()

# 2) Sample–Sample correlation clustermap (top-K variable genes) 
var = df[gene_cols].var(ddof=0).sort_values(ascending=False)
keep = var.head(TOPK_CLUSTermap_GENES).index.tolist()
mat = df[keep].copy()

# correlation across samples
corr = mat.T.corr()

# colors for group annotations
cats = df[GROUP_COL].astype(str)
palette = sns.color_palette("tab20", n_colors=cats.nunique())
col_map = dict(zip(cats.unique(), palette))
col_colors = cats.map(col_map)

g = sns.clustermap(
    corr,
    cmap="viridis",
    row_colors=col_colors,
    col_colors=col_colors,
    figsize=(10, 10),
    xticklabels=False, yticklabels=False
)
g.fig.suptitle(
    f"Sample–Sample Correlation (Top-{TOPK_CLUSTermap_GENES} genes) with {GROUP_COL} colors",
    y=1.02
)
g.savefig(OUT_DIR / f"5_sample_correlation_clustermap_top{TOPK_CLUSTermap_GENES}_by_{GROUP_COL}.png",
          dpi=150, bbox_inches="tight")
plt.close()

# 3) Up vs Down histogram (all genes) 
gdf = df[gene_cols]
up = gdf[gdf > 0].stack()
down = gdf[gdf < 0].stack()

plt.figure(figsize=(10, 6))
sns.histplot(up, bins=50, kde=True, label="Up-Regulated")
sns.histplot(down, bins=50, kde=True, label="Down-Regulated", color="red")
plt.legend()
plt.title("Distribution of Up- vs Down-Regulated Gene Expressions")
plt.xlabel("Differential Expression Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_up_vs_down.png", dpi=150)
plt.close()

# 4) Variance across genes 
var_s = gdf.var(ddof=0)
plt.figure(figsize=(10, 4))
plt.plot(var_s.values)
plt.title("Variance Across Gene Columns")
plt.xlabel("Gene index")
plt.ylabel("Variance")
plt.tight_layout()
plt.savefig(OUT_DIR / "variance_across_genes.png", dpi=150)
plt.close()


