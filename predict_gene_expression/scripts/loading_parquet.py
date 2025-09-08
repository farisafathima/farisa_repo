
from pathlib import Path
import pandas as pd


IN_PATHS = [
    r"C:\Users\faris\SEM_III_PROJECT\PROJECT\de_train.parquet",
]
OUT_DIR = Path(r"C:\Users\faris\SEM_III_PROJECT\PROJECT\csv_out")


OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_parquet(fp: Path) -> pd.DataFrame:
    return pd.read_parquet(fp, engine="pyarrow")

def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

for p in IN_PATHS:
    src = Path(p)
    if not src.exists():
        raise FileNotFoundError(src)
    df = load_parquet(src)
    print(f"[load] {src.name} shape={df.shape}")
    out_csv = OUT_DIR / (src.stem + ".csv")
    save_csv(df, out_csv)
    print(f"[save] {out_csv}")
