from sklearn.metrics import roc_curve, auc
import nibabel as nib
import random
import os
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)


# Load smri scans
def load_mri_img(base_dir):
    all_data = []
    labels = []
    subject_ids = []

    for subject_folder in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_folder, 'anat')
        if not os.path.isdir(subject_path):
            continue

        t1w_file = [f for f in os.listdir(subject_path) if 'space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz' in f]
        mask_file = [f for f in os.listdir(subject_path) if 'space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

        if t1w_file and mask_file:
            t1w_path = os.path.join(subject_path, t1w_file[0])
            mask_path = os.path.join(subject_path, mask_file[0])

            t1w_data = nib.load(t1w_path).get_fdata()
            mask_data = nib.load(mask_path).get_fdata()

            brain_data = (t1w_data * (mask_data > 0)).astype(np.float32)
            label = 0 if 'control' in subject_folder.lower() else 1

            all_data.append(brain_data)
            labels.append(label)
            subject_ids.append(subject_folder)

    if not all_data:
        raise ValueError(f"No valid data found in {base_dir}")

    return all_data, labels, subject_ids


#Load rs-fmri processed scans (schaefer atlas)
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
        else:
            print(f"[Missing] Files not found for subject: {subject_id}")

    return all_corr_matrices, all_timeseries, labels, subject_ids


def sync_modalities(smri_data, smri_labels, smri_ids, fmri_corr, fmri_ts, fmri_ids):
    common_ids = set(smri_ids).intersection(set(fmri_ids))

    synced_smri_data = []
    synced_smri_labels = []
    synced_fmri_corr = []
    synced_fmri_ts = []
    synced_ids = []

    for sid in common_ids:
        i_s = smri_ids.index(sid)
        i_f = fmri_ids.index(sid)

        synced_smri_data.append(smri_data[i_s])
        synced_smri_labels.append(smri_labels[i_s])
        synced_fmri_corr.append(fmri_corr[i_f])
        synced_fmri_ts.append(fmri_ts[i_f])
        synced_ids.append(sid)

    return synced_smri_data, synced_fmri_corr, synced_fmri_ts, synced_smri_labels


#ResNet Branch
class ResNet3D_ASD(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from monai.networks.nets import resnet18

        # Load MONAI 3D ResNet18 backbone
        self.base = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,
            pretrained=pretrained,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True
        )

        # Freeze all, unfreeze selected layers
        for param in self.base.parameters():
            param.requires_grad = False
        for name, module in self.base.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in module.parameters():
                    param.requires_grad = True

        # Feature extractor up to penultimate block
        self.features = nn.Sequential(*list(self.base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, return_feature=True):
        # Input x: [B, 1, 128, 128, 128]
        x = self.features(x)
        x = self.pool(x)
        feat = x.view(x.size(0), -1)   # → [B, 256]
        print("ResNet3D output:", feat.shape)


        if return_feature:
            return feat
        else:
            return self.base.fc(feat)  # → [B, 2]



class BrainNetCNN(nn.Module):
    def __init__(self, num_nodes=100, e2e_filters=32, e2n_filters=64, n2g_filters=128, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.num_nodes = num_nodes

        # Edge-to-Edge (E2E) convolution
        self.e2econv = nn.Conv2d(1, e2e_filters, kernel_size=(1, num_nodes), bias=True)

        # Edge-to-Node (E2N) convolution
        self.e2nconv = nn.Conv2d(e2e_filters, e2n_filters, kernel_size=(num_nodes, 1), bias=True)

        # Node-to-Graph (N2G) layers
        self.fc1 = nn.Linear(e2n_filters, n2g_filters)
        self.fc2 = nn.Linear(n2g_filters, num_classes)

    def forward(self, x, return_feature=False):
        # Input: x [B, 1, N, N]
        x = F.relu(self.e2econv(x))      # → [B, e2e_filters, N, 1]
        x = F.relu(self.e2nconv(x))      # → [B, e2n_filters, 1, 1]
        x = x.view(x.size(0), -1)        # → [B, e2n_filters]
        features = F.relu(self.fc1(x))   # → [B, n2g_filters]
        if return_feature:
            return features              # [B, 128]
        out = self.fc2(features)         # → [B, num_classes]
        return out

#============================= temporal avg pooling ============================#
#
# class TimeSeriesTransformer(nn.Module):
#     def __init__(self, num_rois=100, seq_len=200, d_model=64, nhead=4, num_layers=3, num_classes=2):
#         super(TimeSeriesTransformer, self).__init__()
#         self.input_proj = nn.Linear(num_rois, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         self.fc = nn.Sequential(
#             nn.Linear(d_model, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x, return_feature=False):
#         # Input x: [B, seq_len, num_rois]
#         x = self.input_proj(x)
#         x = self.encoder(x)
#         pooled = x.mean(dim=1)
#         if return_feature:
#             return pooled
#         out = self.fc(pooled)
#         return out



#============================= cls token ============================#
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_rois=100, seq_len=200, d_model=64, nhead=4, num_layers=3, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(num_rois, d_model)

        # [CLS] token as a learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_feature=False):
        x = self.input_proj(x)  # [B, T, d_model]

        # Add [CLS] token to beginning
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x_encoded = self.encoder(x)
        cls_out = x_encoded[:, 0, :]

        if return_feature:
            return cls_out
        else:
            return self.fc(cls_out)



class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities=3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))
        self.fc_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)])

    def forward(self, embeds):
        stacked = torch.stack([proj(e) for proj, e in zip(self.fc_proj, embeds)], dim=1)
        query = self.query.expand(stacked.size(0), -1).unsqueeze(1)
        attn_weights = torch.softmax((query @ stacked.transpose(1, 2)) / np.sqrt(stacked.size(-1)), dim=-1)
        fused = (attn_weights @ stacked).squeeze(1)
        return fused


#=================== cross attention ========================
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, embed_dim, num_heads=4):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#
#     def forward(self, query_embed, context_embeds):
#         # query_embed: [B, D], context_embeds: [B, N_ctx, D]
#         q = query_embed.unsqueeze(1)        # [B, 1, D]
#         k = v = context_embeds              # [B, N_ctx, D]
#         attn_output, _ = self.cross_attn(q, k, v)
#         return attn_output.squeeze(1)       # [B, D]

class MultimodalClassifier(nn.Module):
    def __init__(self, d_emb=64):
        super().__init__()

        # Modality branches
        self.sMRI_branch = ResNet3D_ASD()                   # Outputs: [B, 256]
        self.FC_branch = BrainNetCNN()                      # Features: [B, 128]
        self.DFC_branch = TimeSeriesTransformer(d_model=d_emb)  # Features: [B, 64]

        # Project each modality to common embedding dim
        self.proj_sMRI = nn.Sequential(
            nn.Linear(256, d_emb),
            nn.Dropout(0.3)
        )
        self.proj_FC = nn.Sequential(
            nn.Linear(128, d_emb),
            nn.Dropout(0.3)
        )
        self.proj_DFC = nn.Sequential(
            nn.Linear(64, d_emb),
            nn.Dropout(0.3)
        )

        # Fusion
        self.fusion = AttentionFusion(embed_dim=d_emb)

        # =================== cross attention ========================

        # With this
        # self.fusion = CrossAttentionFusion(embed_dim=d_emb, num_heads=4)







        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_emb, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, smri, fc, dfc):
        # Extract features from each branch
        x1 = self.sMRI_branch(smri, return_feature=True)  # [B, 512]
        x2 = self.FC_branch(fc, return_feature=True)      # [B, 128]
        x3 = self.DFC_branch(dfc, return_feature=True)    # [B, 64]

        # Project features to common dim
        x1_proj = self.proj_sMRI(x1)                      # [B, d_emb]
        x2_proj = self.proj_FC(x2)                        # [B, d_emb]
        x3_proj = self.proj_DFC(x3)                       # [B, d_emb]

        # Attention-based fusion
        fused = self.fusion([x1_proj, x2_proj, x3_proj])  # [B, d_emb]

        # =================== cross attention ========================

        # # With this
        # context = torch.stack([x1_proj, x2_proj], dim=1)  # [B, 2, d_emb]
        # fused = self.fusion(x3_proj, context)
        # =================== cross attention ========================





        # Final classification
        out = self.classifier(fused)                      # [B, 2]
        return out

class MultimodalDataset(Dataset):
    def __init__(self, smri_data, fc_data, dfc_data, labels, target_shape=(128, 128, 128), seq_len=None):
        self.smri = smri_data
        self.fc = fc_data
        self.dfc = dfc_data
        self.labels = labels
        self.target_shape = target_shape
        self.seq_len = seq_len or max(x.shape[0] for x in dfc_data)  # default: max length in dfc_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch.nn.functional as F

        x1 = torch.tensor(self.smri[idx]).unsqueeze(0).float()
        x2 = torch.tensor(self.fc[idx]).unsqueeze(0).float()
        x3 = torch.tensor(self.dfc[idx]).float()
        y = torch.tensor(self.labels[idx]).long()

        # Resize sMRI
        x1 = F.interpolate(x1.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)

        # Pad x3 to max seq length (time axis)
        T, R = x3.shape
        if T < self.seq_len:
            pad_len = self.seq_len - T
            x3 = F.pad(x3, (0, 0, 0, pad_len))  # pad along time dimension (left at beginning)
        elif T > self.seq_len:
            x3 = x3[:self.seq_len]

        return x1, x2, x3, y



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for smri, fc, dfc, label in loader:
        smri, fc, dfc, label = smri.to(device), fc.to(device), dfc.to(device), label.to(device)
        output = model(smri, fc, dfc)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
        correct += (output.argmax(dim=1) == label).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)




def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for smri, fc, dfc, label in loader:
            smri, fc, dfc, label = smri.to(device), fc.to(device), dfc.to(device), label.to(device)
            output = model(smri, fc, dfc)
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


def cross_validate_model(smri_data, fc_data, dfc_data, labels, skf=5, epochs=100, lr=1e-5, wd=1e-4, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(ts.shape[0] for ts in dfc_data)
    dataset = MultimodalDataset(smri_data, fc_data, dfc_data, labels, seq_len=max_seq_len)
    skf = StratifiedKFold(n_splits=skf, shuffle=True, random_state=42)

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    roc_curves, aucs = [], []
    all_labels, all_probs, all_preds = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        set_seed(42 + fold)
        print(f"\n=== Fold {fold + 1} ===")
        fold_train_accs, fold_val_accs = [], []
        model = MultimodalClassifier().to(device)



        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)



        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_prec, val_rec, val_auc, val_probs, val_labels, val_preds = evaluate(
                model, val_loader, criterion, device)

            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)


            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch + 1:03d}] "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                      f"F1: {val_f1:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)


        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        fold_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        aucs.append(fold_auc)
        all_labels.extend(val_labels)
        all_probs.extend(val_probs)
        all_preds.extend(val_preds)


        plot_metrics(
            fold_train_accs,
            fold_val_accs,
            train_losses[-len(fold_train_accs):],
            val_losses[-len(fold_val_accs):],
            save_path=f"metrics_d_fold{fold + 1}.png"
        )


        train_accs.extend(fold_train_accs)
        val_accs.extend(fold_val_accs)

    return train_accs, val_accs, train_losses, val_losses, roc_curves, aucs, all_labels, all_probs, all_preds, model




def plot_metrics(train_accs, val_accs, train_losses, val_losses, save_path="metrics_d.png"):
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(6, 5))

    # Accuracy Plot Only
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_roc_curves(roc_curves, aucs, save_path="roc_curves_d.png"):
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





def plot_confusion_matrix(labels, predictions, class_names, save_path="confusion_matrix_d.png"):

    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved as {save_path}")

class_names = ["Control", "Patient"]






if __name__ == "__main__":

    set_seed(42)
    # Load sMRI
    # smri_data, smri_labels, smri_ids = load_mri_img("./data")
    smri_data, smri_labels, smri_ids = load_mri_img("/mnt/data/shyam/farisa/ASD_proj/data/fmriprep")

    # Load fMRI
    # fc_data, ts_data, fmri_labels, fmri_ids = load_fmri_schaefer("./abide")
    fc_data, ts_data, fmri_labels, fmri_ids = load_fmri_schaefer("/mnt/data/shyam/farisa/ASD_proj/data/abide")

    # Sync data across modalities
    synced_smri, synced_fc, synced_dfc, synced_labels = sync_modalities(
        smri_data, smri_labels, smri_ids,
        fc_data, ts_data, fmri_ids
    )

    # Train the model using cross-validation
    train_accs, val_accs, train_losses, val_losses, roc_curves, aucs, all_labels, all_probs, all_preds, model = cross_validate_model(
        smri_data=synced_smri,
        fc_data=synced_fc,
        dfc_data=synced_dfc,
        labels=synced_labels,
        skf=5,
        epochs = 150,
        lr = 1e-5,
        wd = 1e-4,
        batch_size = 64
    )

    print(f"Total samples in dataset: {len(synced_smri)}")


    plot_roc_curves(roc_curves, aucs)

    # Plot confusion matrix
    class_names = ["Control", "Patient"]
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path="confusion_matrix_d.png")
