import os
import nibabel as nib
import torch.nn.functional as F
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = None

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
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


def load_mri_img(base_dir):
    data = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        t1w_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz' in f]
        mask_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

        if t1w_file and mask_file:
            t1w_img_path = os.path.join(root, t1w_file[0])
            mask_img_path = os.path.join(root, mask_file[0])

            # Load images
            t1w_img = nib.load(t1w_img_path)
            t1w_data = t1w_img.get_fdata()

            mask_img = nib.load(mask_img_path)
            mask_data = mask_img.get_fdata()

            # Apply brain mask
            brain_data = t1w_data * (mask_data > 0)
            brain_data = brain_data.astype(np.float32)

            brain_data = np.expand_dims(brain_data, axis=0)  # Add channel dimension

            data.append(brain_data)

            # Assign label based on directory name
            label = 0 if 'control' in root else 1
            labels.append(label)

    if not data:
        raise ValueError(f"No data found in {base_dir}. Check your directory structure and file paths.")

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Convert to Torch tensors with correct shape (C, D, H, W)
    data = torch.tensor(data)

    print(f"Loaded {len(data)} images and {len(labels)} labels. Shape: {data.shape}")

    return data, torch.tensor(labels)


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
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc




base_dir = r'/mnt/data/shyam/farisa/ASD_proj/data/fmriprep'
data, labels = load_mri_img(base_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5)

all_final_acc = []

# loop over folds
for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
    print(f"Starting Fold {fold + 1}")

    train_images, train_labels = data[train_idx], labels[train_idx]
    val_images, val_labels = data[val_idx], labels[val_idx]

    train_loader = DataLoader(TensorDataset(torch.tensor(train_images), torch.tensor(train_labels)), batch_size=32,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_images), torch.tensor(val_labels)), batch_size=32,
                            shuffle=False)

    model = Simple3DCNN(in_channels=1, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if (epoch + 1) % 10 == 0:
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    all_final_acc.append(val_acc)

mean_final_acc = np.mean(all_final_acc)
std_final_acc = np.std(all_final_acc)

print(f'Mean Final Accuracy: {mean_final_acc:.2f}% ± {std_final_acc:.2f}%')

