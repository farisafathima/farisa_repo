"""
Purpose: Add ChemBERTa embeddings for SMILES to de_train.csv.
- Batches unique SMILES for speed.
- Merges embeddings back as columns: embedding_0 ... embedding_(D-1).
"""

from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ====== EDIT THESE ======
IN_CSV  = Path(r"C:\Users\faris\SEM_III_PROJECT\PROJECT\csv_out\de_train.csv")
OUT_CSV = Path(r"C:\Users\faris\SEM_III_PROJECT\PROJECT\de_train_enriched.csv")
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================

def build_chemberta(smiles_series: pd.Series,
                    model_name: str = MODEL_NAME,
                    batch_size: int = BATCH_SIZE,
                    device: str = DEVICE) -> pd.DataFrame:

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    # Map unique SMILES → embedding
    uniq = smiles_series.dropna().astype(str).unique().tolist()
    cache = {}

    with torch.no_grad():
        for i in range(0, len(uniq), batch_size):
            batch = uniq[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            out = mdl(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            cache.update({s: e for s, e in zip(batch, emb)})

    # Reconstruct in original row order
    mat = [cache.get(str(s), None) for s in smiles_series]
    if any(v is None for v in mat):
        raise ValueError("Some SMILES had no embedding (unexpected).")
    emb_df = pd.DataFrame(mat)
    emb_df.columns = [f"embedding_{i}" for i in range(emb_df.shape[1])]
    emb_df.index = smiles_series.index
    return emb_df

# ---- Run ----
df = pd.read_csv(IN_CSV)
if "SMILES" not in df.columns:
    raise KeyError("Expected a 'SMILES' column in input CSV.")

print(f"[input] {IN_CSV.name} shape={df.shape}")
emb_df = build_chemberta(df["SMILES"])
df_enriched = pd.concat([df.drop(columns=["SMILES"]), df["SMILES"], emb_df], axis=1)

print(f"[enriched] shape={df_enriched.shape} (added {emb_df.shape[1]} embedding cols)")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_enriched.to_csv(OUT_CSV, index=False)
print(f"[save] {OUT_CSV}")
