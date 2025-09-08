import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_mri_img(base_dir):
    all_t1w_data = []
    all_masks = []
    all_data = []
    labels = []
    file_paths = []  

    for root, dirs, files in os.walk(base_dir):
        t1w_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz' in f]
        mask_file = [f for f in files if 'space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz' in f]

        if t1w_file and mask_file:
            t1w_img_path = os.path.join(root, t1w_file[0])
            mask_img_path = os.path.join(root, mask_file[0])

            t1w_img = nib.load(t1w_img_path)
            t1w_data = t1w_img.get_fdata()

            mask_img = nib.load(mask_img_path)
            mask_data = mask_img.get_fdata()

            # Apply brain mask
            brain_data = t1w_data * (mask_data > 0)
            brain_data = brain_data.astype(np.float32)

            all_t1w_data.append(t1w_data)
            all_masks.append(mask_data)
            all_data.append(brain_data)
            file_paths.append(t1w_img_path)


            label = 0 if 'control' in root else 1
            labels.append(label)

    if not all_data:
        raise ValueError(f"No data found in {base_dir}. Check your directory structure and file paths.")

    return all_t1w_data, all_masks, all_data, labels, file_paths


def plot_single_subject(t1w_data, mask_data, brain_data, subject_id):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate mid-slices
    mid_slice = t1w_data.shape[2] // 2

    # Plot T1w slice
    axes[0].imshow(t1w_data[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0].set_title('T1w Slice')
    axes[0].axis('off')

    # Plot brain mask slice
    axes[1].imshow(mask_data[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[1].set_title('Brain Mask Slice')
    axes[1].axis('off')

    # Plot brain region slice
    axes[2].imshow(brain_data[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[2].set_title('Brain Region Slice')
    axes[2].axis('off')

    plt.suptitle(f'Subject ID: {subject_id}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_mid_slices(data, subject_ids):
    num_subjects = len(data)
    fig, axes = plt.subplots(num_subjects, 4, figsize=(10, 3 * num_subjects))

    if num_subjects == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (brain_data, subject_id) in enumerate(zip(data, subject_ids)):
        mid_sagittal = brain_data[brain_data.shape[0] // 2, :, :]
        mid_coronal = brain_data[:, brain_data.shape[1] // 2, :]
        mid_axial = brain_data[:, :, brain_data.shape[2] // 2]

        axes[i, 0].text(0.5, 0.5, subject_id, fontsize=10, ha='center', va='center', wrap=True)
        axes[i, 0].axis('off')

        # Plot sagittal
        axes[i, 1].imshow(mid_sagittal.T, cmap='gray', origin='lower')
        axes[i, 1].axis('off')

        # Plot coronal
        axes[i, 2].imshow(mid_coronal.T, cmap='gray', origin='lower')
        axes[i, 2].axis('off')

        # Plot axial
        axes[i, 3].imshow(mid_axial.T, cmap='gray', origin='lower')
        axes[i, 3].axis('off')

        if i == 0:
            axes[i, 1].set_title('Sagittal', fontsize=12)
            axes[i, 2].set_title('Coronal', fontsize=12)
            axes[i, 3].set_title('Axial', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def print_shapes(all_t1w_data, all_data, all_masks, subject_ids):
    for subject_id, t1w_data, mask_data, brain_data in zip(subject_ids, all_t1w_data, all_masks, all_data):
        print(f"Subject {subject_id}:")
        print(f"  T1w image shape: {t1w_data.shape}")
        print(f"  Brain mask shape: {mask_data.shape}")
        print(f"  Brain region shape: {brain_data.shape}")
        print(f"  Minimum voxel value of brain region: {np.min(brain_data):.2f}")
        print(f"  Maximum voxel value of brain region: {np.max(brain_data):.2f}\n")


def robust_zscore_normalize(data):
    normalized_data = []
    for brain_data in data:
        median = np.median(brain_data[brain_data > 0])  # Median of non-zero voxels
        iqr = np.percentile(brain_data[brain_data > 0], 75) - np.percentile(brain_data[brain_data > 0], 25)  # IQR
        normalized_brain_data = (brain_data - median) / iqr  # formula
        normalized_data.append(normalized_brain_data)
    return normalized_data



base_dir = r'/mnt/data/shyam/farisa/ASD_proj/data/fmriprep'
all_t1w_data, all_masks, all_data, labels, file_paths = load_mri_img(base_dir)


subject_ids = [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path)))) for path in file_paths]

# Print shapes of images and brain statistics
print_shapes(all_t1w_data, all_data, all_masks, subject_ids)

# Plot T1w slice, brain mask slice, and brain region slice for a single subject
plot_single_subject(all_t1w_data[0], all_masks[0], all_data[0], subject_ids[0])

# Plot mid-slices of sagittal, coronal, and axial views for all subjects
plot_mid_slices(all_data, subject_ids)

# Split data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.2, random_state=42)

# Normalize training and test sets separately
train_data_normalized = robust_zscore_normalize(train_data)
test_data_normalized = robust_zscore_normalize(test_data)

# Print normalized voxel ranges for verification
print("Training set normalized voxel range:")
print(f"  Min: {np.min(train_data_normalized):.2f}, Max: {np.max(train_data_normalized):.2f}")

print("Test set normalized voxel range:")
print(f"  Min: {np.min(test_data_normalized):.2f}, Max: {np.max(test_data_normalized):.2f}")


# output
# Subject sub-control50037:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -18.67
#   Maximum voxel value of brain region: 1524.33
#
# Subject sub-control50038:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -19.21
#   Maximum voxel value of brain region: 2003.60
#
# Subject sub-control50436:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -28.47
#   Maximum voxel value of brain region: 1327.34
#
# Subject sub-patient50027:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -40.67
#   Maximum voxel value of brain region: 2111.35
#
# Subject sub-patient51166:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -440.61
#   Maximum voxel value of brain region: 22044.67
#
# Subject sub-patient51218:
#   T1w image shape: (97, 115, 97)
#   Brain mask shape: (97, 115, 97)
#   Brain region shape: (97, 115, 97)
#   Minimum voxel value of brain region: -6.03
#   Maximum voxel value of brain region: 723.05
#
# Training set normalized voxel range:
#   Min: -3.16, Max: 18.04
# Test set normalized voxel range:
#   Min: -3.04, Max: 12.82
#
# Process finished with exit code 0
