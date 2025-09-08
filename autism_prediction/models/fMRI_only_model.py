import os
import scipy.io as sio
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data Loading for fMRI (Correlation Matrix + Time Series)

def load_fmri_schaefer(base_dir):
    all_corr_matrices = []
    all_timeseries = []
    labels = []
    subject_ids = []

    for subject_folder in os.listdir(base_dir):
        subj_path = os.path.join(base_dir, subject_folder)
        if not os.path.isdir(subj_path):
            continue

        label = 0 if 'control' in subject_folder.lower() else 1
        subject_id = subject_folder

        corr_file = f"{subject_id}_schaefer100_correlation_matrix.mat"
        ts_file = f"{subject_id}_schaefer100_features_timeseries.mat"

        corr_path = os.path.join(subj_path, corr_file)
        ts_path = os.path.join(subj_path, ts_file)

        if os.path.exists(corr_path) and os.path.exists(ts_path):
            try:
                corr_data = sio.loadmat(corr_path)
                ts_data = sio.loadmat(ts_path)

                corr_matrix = next(v for k, v in corr_data.items() if not k.startswith('__'))
                timeseries = next(v for k, v in ts_data.items() if not k.startswith('__'))

                all_corr_matrices.append(corr_matrix.astype(np.float32))
                all_timeseries.append(timeseries.astype(np.float32))
                labels.append(label)
                subject_ids.append(subject_id)
            except Exception as e:
                print(f"[Warning] Failed to load {subject_id}: {e}")

    if not all_corr_matrices:
        raise ValueError(f"No valid data found in {base_dir}")

    return all_corr_matrices, all_timeseries, labels, subject_ids


#Dataset definition

class MultimodalDataset(Dataset):
    def __init__(self, fc_data, dfc_data, labels, seq_len=None):
        self.fc = fc_data
        self.dfc = dfc_data
        self.labels = labels
        self.seq_len = seq_len or max(x.shape[0] for x in dfc_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch.nn.functional as F

        x1 = torch.tensor(self.fc[idx]).unsqueeze(0).float()  # [1, 100, 100]
        x2 = torch.tensor(self.dfc[idx]).float()              # [T, 100]
        y = torch.tensor(self.labels[idx]).long()

        # Pad x2 to max seq length (time axis)
        T, R = x2.shape
        if T < self.seq_len:
            pad_len = self.seq_len - T
            x2 = F.pad(x2, (0, 0, 0, pad_len))
        elif T > self.seq_len:
            x2 = x2[:self.seq_len]

        return x1, x2, y


# BrainNetCNN for Correlation Matrix


class BrainNetCNN(nn.Module):
    def __init__(self, num_nodes=100, e2e_filters=32, e2n_filters=64, n2g_filters=128, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.e2econv = nn.Conv2d(1, e2e_filters, kernel_size=(1, num_nodes), bias=True)
        self.e2nconv = nn.Conv2d(e2e_filters, e2n_filters, kernel_size=(num_nodes, 1), bias=True)
        self.fc1 = nn.Linear(e2n_filters, n2g_filters)
        self.fc2 = nn.Linear(n2g_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.e2econv(x))
        x = F.relu(self.e2nconv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


# Transformer for Time Series Matrix


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_rois=100, seq_len=200, d_model=64, nhead=4, num_layers=3, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(num_rois, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        #temporal average pooling
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


# Attention-based Fusion Layer


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities=2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))
        self.fc_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)])

    def forward(self, embeds):
        stacked = torch.stack([proj(e) for proj, e in zip(self.fc_proj, embeds)], dim=1)
        query = self.query.expand(stacked.size(0), -1).unsqueeze(1)
        attn_weights = torch.softmax((query @ stacked.transpose(1, 2)) / np.sqrt(stacked.size(-1)), dim=-1)
        fused = (attn_weights @ stacked).squeeze(1)
        return fused


# Combined Model


class MultimodalClassifier(nn.Module):
    def __init__(self, d_emb=64):
        super().__init__()
        self.FC_branch = BrainNetCNN()
        self.DFC_branch = TimeSeriesTransformer(d_model=d_emb)

        self.fusion = AttentionFusion(embed_dim=d_emb)
        self.proj_FC = nn.Sequential(nn.Linear(2, d_emb))
        self.proj_DFC = nn.Sequential(nn.Linear(2, d_emb))

        self.classifier = nn.Sequential(
            nn.Linear(d_emb, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, fc, dfc):
        x1 = self.FC_branch(fc)
        x2 = self.DFC_branch(dfc)

        x1_proj = self.proj_FC(x1)
        x2_proj = self.proj_DFC(x2)

        fused = self.fusion([x1_proj, x2_proj])
        out = self.classifier(fused)
        return out




# Training


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for fc, dfc, label in loader:
        fc, dfc, label = fc.to(device), dfc.to(device), label.to(device)

        output = model(fc, dfc)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        correct += (output.argmax(dim=1) == label).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# Evaluation


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for fc, dfc, label in loader:
            fc, dfc, label = fc.to(device), dfc.to(device), label.to(device)

            output = model(fc, dfc)
            loss = criterion(output, label)

            total_loss += loss.item() * label.size(0)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            labels = label.cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)

            correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = float('nan')

    return total_loss / len(loader.dataset), acc, f1, prec, rec, auc_score, all_probs, all_labels, all_preds


# Cross-Validation Training Loop


def cross_validate_model(fc_data, dfc_data, labels, skf=5, epochs=100, lr=1e-5, wd=1e-4, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(ts.shape[0] for ts in dfc_data)
    dataset = MultimodalDataset(fc_data, dfc_data, labels, seq_len=max_seq_len)
    skf = StratifiedKFold(n_splits=skf, shuffle=True, random_state=42)

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    roc_curves, aucs = [], []
    all_labels, all_probs, all_preds = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Fold {fold + 1} ===")
        fold_train_accs, fold_val_accs = [], []

        model = MultimodalClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, num_workers=4)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_prec, val_rec, val_auc, val_probs, val_labels, val_preds = evaluate(
                model, val_loader, criterion, device)

            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch + 1:03d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
                      f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")

        # Collect ROC and AUC after fold
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        fold_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        aucs.append(fold_auc)
        all_labels.extend(val_labels)
        all_probs.extend(val_probs)
        all_preds.extend(val_preds)

        train_accs.extend(fold_train_accs)
        val_accs.extend(fold_val_accs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_accs, val_accs, train_losses, val_losses, roc_curves, aucs, all_labels, all_probs, all_preds


# Plotting Functions


def plot_metrics(train_accs, val_accs, train_losses, val_losses, save_path="metrics.png"):
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(roc_curves, aucs, save_path="roc_curves.png"):
    mean_tpr = np.mean([np.interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in roc_curves], axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure(figsize=(10, 7))
    for i, (fpr, tpr) in enumerate(roc_curves):
        plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} AUC: {aucs[i]:.3f}")

    plt.plot(np.linspace(0, 1, 100), mean_tpr, label=f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}", color='blue', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(labels, predictions, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300)
    plt.close()


# Main Execution


if __name__ == "__main__":
    set_seed(42)

    # Load fMRI Data
    base_dir = "/mnt/data/shyam/farisa/ASD_proj/data/fMRI_data"
    fc_data, ts_data, labels, _ = load_fmri_schaefer(base_dir)

    # Train the model using cross-validation
    train_accs, val_accs, train_losses, val_losses, roc_curves, aucs, all_labels, all_probs, all_preds = cross_validate_model(
        fc_data=fc_data,
        dfc_data=ts_data,
        labels=labels,
        skf=5,
        epochs=150,
        lr=1e-5,
        wd=1e-4,
        batch_size=32
    )

    # Plot Metrics
    plot_metrics(train_accs, val_accs, train_losses, val_losses)
    plot_roc_curves(roc_curves, aucs)
    plot_confusion_matrix(all_labels, all_preds, ["Control", "Patient"])
