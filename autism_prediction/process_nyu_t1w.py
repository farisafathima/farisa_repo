import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(42)
np.random.seed(42)

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = None
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, 2).to(x.device)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def load_mri_img(base_dir, allowed_ids=None):
    import re
    data = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        if not folder_name.startswith('sub-') or folder_name.endswith('.html'):
            continue
        match = re.search(r'\d+', folder_name)
        if not match:
            continue
        numeric_id = match.group(0)
        if allowed_ids is not None and numeric_id not in allowed_ids:
            continue
        anat_path = os.path.join(root, 'anat')
        if not os.path.isdir(anat_path):
            continue
        t1w_file = None
        mask_file = None
        for f in os.listdir(anat_path):
            if 'space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz' in f:
                t1w_file = os.path.join(anat_path, f)
            if 'space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f:
                mask_file = os.path.join(anat_path, f)
        if not t1w_file or not mask_file:
            continue
        t1w_img = nib.load(t1w_file)
        t1w_data = t1w_img.get_fdata()
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        brain_data = t1w_data * (mask_data > 0)
        brain_data = brain_data.astype(np.float32)
        brain_data = np.expand_dims(brain_data, axis=0)
        data.append(brain_data)
        label = 0 if 'control' in folder_name else 1
        labels.append(label)
    if not data:
        raise ValueError(f"No valid MRI data found in {base_dir} for given IDs.")
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return torch.tensor(data), torch.tensor(labels)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_metrics(history, metric_name, filename):
    epochs = range(1, len(history['train']) + 1)
    plt.figure()
    plt.plot(epochs, history['train'], label='Train')
    plt.plot(epochs, history['val'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_roc_curve(y_true, y_score, filename):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename, format='pdf')
    plt.close()
    return roc_auc

def run_experiment(params, train_images, train_labels, val_images, val_labels, device, fold_id=1, model_id=0):
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=params["batch_size"], shuffle=False)
    model = Simple3DCNN(in_channels=1, num_classes=2, dropout_rate=params["dropout_rate"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(params["num_epochs"]):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_labels_all, val_preds_all, val_probs_all = validate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{params['num_epochs']}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
    final_train_loss = history['train_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_acc'][-1]
    cm = confusion_matrix(val_labels_all, val_preds_all)
    report = classification_report(val_labels_all, val_preds_all, output_dict=True)
    f1 = f1_score(val_labels_all, val_preds_all, average='weighted')
    precision = precision_score(val_labels_all, val_preds_all, average='weighted')
    recall = recall_score(val_labels_all, val_preds_all, average='weighted')
    roc_auc = plot_roc_curve(val_labels_all, val_probs_all, f"model{model_id}_fold{fold_id}_roc.pdf")
    plot_metrics({'train': history['train_loss'], 'val': history['val_loss']}, 'Loss', f"model{model_id}_fold{fold_id}_loss.pdf")
    plot_metrics({'train': history['train_acc'], 'val': history['val_acc']}, 'Accuracy', f"model{model_id}_fold{fold_id}_accuracy.pdf")
    fold_metrics = {
        'train_loss': final_train_loss,
        'train_acc': final_train_acc,
        'val_loss': final_val_loss,
        'val_acc': final_val_acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'history': history
    }
    return fold_metrics

def main():
    base_dir = r'/mnt/data/shyam/farisa/ASD_proj/data/fmriprep'
    nyu_ids_df = pd.read_csv(r'/mnt/data/shyam/farisa/ASD_proj/data/nyu_common_subjects.txt', sep='\t')
    allowed_ids = set(nyu_ids_df['SUB_ID'].astype(str))
    data, labels = load_mri_img(base_dir, allowed_ids=allowed_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(data, labels))
    train_images, train_labels = data[train_idx], labels[train_idx]
    val_images, val_labels = data[val_idx], labels[val_idx]
    params = {
        "learning_rate": 0.00001,
        "weight_decay": 0.0001,
        "dropout_rate": 0.3,
        "batch_size": 32,
        "num_epochs": 100
    }
    print(f"\nUsing fixed hyperparameters: {params}")
    fold_metrics = run_experiment(
        params,
        train_images, train_labels,
        val_images, val_labels,
        device,
        fold_id=1,
        model_id=1
    )
    print(f"\nFinal Results with fixed hyperparameters:")
    print(f"Average Final Training Accuracy: {fold_metrics['train_acc']:.2f}%")
    print(f"Average Final Validation Accuracy: {fold_metrics['val_acc']:.2f}%")
    print(f"Average Weighted F1: {fold_metrics['f1']:.4f}")
    print(f"Average Precision: {fold_metrics['precision']:.4f}")
    print(f"Average Recall: {fold_metrics['recall']:.4f}")
    print(f"Average ROC AUC: {fold_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
