import os
import nibabel as nib
import torch.nn.functional as F
import torchio as tio
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import random
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from monai.networks.nets import resnet
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# def augment_data(tensor):
#     transform = tio.Compose([
#         tio.RandomAffine(scales=(0.95, 1.05), degrees=10, translation=5),
#         tio.RandomGamma(log_gamma=(-0.2, 0.2)),
#         tio.RandomNoise(std=0.01),
#         tio.RandomFlip(axes=(0, 1, 2))
#     ])
#     return transform(tensor)


def load_mri_img(base_dir):
    data, labels = [], []
    for root, dirs, files in os.walk(base_dir):
        t1w_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz' in f]
        mask_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

        if t1w_file and mask_file:
            t1w_img = nib.load(os.path.join(root, t1w_file[0])).get_fdata()
            mask_img = nib.load(os.path.join(root, mask_file[0])).get_fdata()
            brain_data = (t1w_img * (mask_img > 0)).astype(np.float32)
            brain_data = np.expand_dims(brain_data, axis=0)
            data.append(brain_data)
            labels.append(0 if 'control' in root else 1)

    if not data:
        raise ValueError(f"No data found in {base_dir}. Check your directory structure and file paths.")

    data = torch.tensor(np.array(data), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    print(f"Loaded {len(data)} images. Shape: {data.shape}")
    return data, labels


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return 100 * (predicted == labels).sum().item() / labels.size(0)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), 100 * correct / total




# Load data
base_dir = r'/mnt/data/shyam/farisa/ASD_proj/data/fmriprep'
data, labels = load_mri_img(base_dir)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5)
all_final_acc = []
model_name = 'MONAI_ResNet50_3D'

for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
    print(f"Starting Fold {fold + 1}")
    train_images, train_labels = data[train_idx], labels[train_idx]
    val_images, val_labels = data[val_idx], labels[val_idx]

    #train_images = torch.stack([augment_data(img) for img in train_images])
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=32, shuffle=False)

    from monai.networks.nets import resnet

    model = resnet.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2, dropout=0.3).to(device)
   
    for name, param in model.named_parameters():
        if "layer4" in name or name.startswith("fc"):  # Fine-tune last block + classifier
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(100):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        if (epoch + 1) % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    all_final_acc.append(val_acc)

    # Save loss and accuracy plots
    with PdfPages(f'{model_name}_fold{fold + 1}_loss_acc.pdf') as pdf:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Loss (Fold {fold + 1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title(f'Accuracy (Fold {fold + 1})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()



mean_acc = np.mean(all_final_acc)
std_acc = np.std(all_final_acc)

print(f'Mean Final Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
