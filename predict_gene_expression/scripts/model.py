import os, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Config 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

K_TOP = 400
EPOCHS = 100
BATCH_SIZE = 8
PRINT_EVERY = 10

DATA_DIR = Path("./PROJECT/clean")
READY_DIR = Path("./PROJECT/ready")
PLOT_DIR  = READY_DIR / "plots"
READY_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#  1) Load data
X = pd.read_csv(DATA_DIR / "X_features.csv")
Y = pd.read_csv(DATA_DIR / "Y_targets.csv")
assert len(X) == len(Y), "X and Y must have same number of rows"
print(f"[Data] X: {X.shape}  Y: {Y.shape}")

#  2) Prepare data 
#  - 80/20 split
#  - Select Top-K targets by variance on TRAIN only (avoid leakage)
X_train_df, X_val_df, Y_train_df_full, Y_val_df_full = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

var_train = Y_train_df_full.var(axis=0, ddof=0)
top_cols = var_train.sort_values(ascending=False).head(K_TOP).index.tolist()
Y_train_df = Y_train_df_full[top_cols].copy()
Y_val_df   = Y_val_df_full[top_cols].copy()

pd.Series(top_cols, name="gene").to_csv(
    READY_DIR / f"selected_genes_top{K_TOP}_by_var_single_split.csv",
    index=False
)
print(f"[Targets] Selected Top-{K_TOP} by variance on train split.")

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train = imputer.fit_transform(X_train_df.values).astype(np.float32)
X_val   = imputer.transform(X_val_df.values).astype(np.float32)

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)

# Targets -> float32
Y_train = Y_train_df.values.astype(np.float32)
Y_val   = Y_val_df.values.astype(np.float32)

print(f"[Shapes Top-{K_TOP}] X_train:{X_train.shape} X_val:{X_val.shape}  Y_train:{Y_train.shape} Y_val:{Y_val.shape}")

# 3) Models 
XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=1
)
xgb = MultiOutputRegressor(XGBRegressor(**XGB_PARAMS))

class MLPRegressorTorch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def to_sequence(x2d: np.ndarray, steps: int = 8) -> np.ndarray:
    """(B, F) -> (B, steps, F_step) with zero padding if needed."""
    B, F = x2d.shape
    step_feats = int(math.ceil(F / steps))
    newF = steps * step_feats
    if newF != F:
        pad = np.zeros((B, newF - F), dtype=x2d.dtype)
        x2d = np.concatenate([x2d, pad], axis=1)
    return x2d.reshape(B, steps, step_feats)

class GRURegressor(nn.Module):
    def __init__(self, in_step_feats: int, hidden: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_step_feats,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.25 if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, out_dim)
        )
    def forward(self, x_seq):
        out, _ = self.gru(x_seq)
        last = out[:, -1, :]
        return self.head(last)

# 4) Train models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# XGB
t0 = time.time()
xgb.fit(X_train, Y_train)
xgb_time = time.time() - t0

# MLP
mlp = MLPRegressorTorch(in_dim=X_train_scaled.shape[1], out_dim=Y_train.shape[1]).to(DEVICE)
opt_mlp  = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn  = nn.MSELoss()

ds_mlp_tr = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(Y_train))
ds_mlp_va = TensorDataset(torch.tensor(X_val_scaled),   torch.tensor(Y_val))
dl_mlp_tr = DataLoader(ds_mlp_tr, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE=="cuda"))
dl_mlp_va = DataLoader(ds_mlp_va, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE=="cuda"))

mlp_train_losses, mlp_val_losses = [], []
t1 = time.time()
for epoch in range(1, EPOCHS+1):
    # train
    mlp.train(); run_tr = 0.0
    for xb, yb in dl_mlp_tr:
        xb, yb = xb.to(DEVICE, dtype=torch.float32), yb.to(DEVICE, dtype=torch.float32)
        opt_mlp.zero_grad()
        pred = mlp(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_mlp.step()
        run_tr += loss.item() * xb.size(0)
    tr_loss = run_tr / len(ds_mlp_tr)
    mlp_train_losses.append(tr_loss)

    # validate
    mlp.eval(); run_va = 0.0
    with torch.no_grad():
        for xb, yb in dl_mlp_va:
            xb, yb = xb.to(DEVICE, dtype=torch.float32), yb.to(DEVICE, dtype=torch.float32)
            pred = mlp(xb)
            loss = loss_fn(pred, yb)
            run_va += loss.item() * xb.size(0)
    va_loss = run_va / len(ds_mlp_va)
    mlp_val_losses.append(va_loss)

    if epoch % PRINT_EVERY == 0:
        print(f"[MLP] Epoch {epoch:3d}/{EPOCHS}  Train MSE: {tr_loss:.4f}  Val MSE: {va_loss:.4f}")
mlp_time = time.time() - t1

# GRU
STEPS = 8
Xtr_seq = to_sequence(X_train_scaled, steps=STEPS)
Xva_seq = to_sequence(X_val_scaled,   steps=STEPS)
in_step_feats = Xtr_seq.shape[2]

gru = GRURegressor(in_step_feats=in_step_feats, hidden=128, out_dim=Y_train.shape[1]).to(DEVICE)
opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)

ds_gru_tr = TensorDataset(torch.tensor(Xtr_seq), torch.tensor(Y_train))
ds_gru_va = TensorDataset(torch.tensor(Xva_seq), torch.tensor(Y_val))
dl_gru_tr = DataLoader(ds_gru_tr, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE=="cuda"))
dl_gru_va = DataLoader(ds_gru_va, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE=="cuda"))

gru_train_losses, gru_val_losses = [], []
t2 = time.time()
for epoch in range(1, EPOCHS+1):
    gru.train(); run_tr = 0.0
    for xb, yb in dl_gru_tr:
        xb, yb = xb.to(DEVICE, dtype=torch.float32), yb.to(DEVICE, dtype=torch.float32)
        opt_gru.zero_grad()
        pred = gru(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_gru.step()
        run_tr += loss.item() * xb.size(0)
    tr_loss = run_tr / len(ds_gru_tr)
    gru_train_losses.append(tr_loss)

    gru.eval(); run_va = 0.0
    with torch.no_grad():
        for xb, yb in dl_gru_va:
            xb, yb = xb.to(DEVICE, dtype=torch.float32), yb.to(DEVICE, dtype=torch.float32)
            pred = gru(xb)
            loss = loss_fn(pred, yb)
            run_va += loss.item() * xb.size(0)
    va_loss = run_va / len(ds_gru_va)
    gru_val_losses.append(va_loss)

    if epoch % PRINT_EVERY == 0:
        print(f"[GRU] Epoch {epoch:3d}/{EPOCHS}  Train MSE: {tr_loss:.4f}  Val MSE: {va_loss:.4f}")
gru_time = time.time() - t2




#5) Visualisation
def plot_learning_curve(losses, val_losses, title, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses)+1), losses, marker='o', label='Train MSE')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Val MSE')
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(outpath); plt.close()

plot_learning_curve(
    mlp_train_losses, mlp_val_losses,
    f"MLP Learning Curve (Train/Val MSE) — Top {K_TOP}",
    PLOT_DIR / f"mlp_learning_curve_top{K_TOP}.png"
)
plot_learning_curve(
    gru_train_losses, gru_val_losses,
    f"GRU Learning Curve (Train/Val MSE) — Top {K_TOP}",
    PLOT_DIR / f"gru_learning_curve_top{K_TOP}.png"
)

print(f"\nSaved artifacts and plots to: {READY_DIR}")
