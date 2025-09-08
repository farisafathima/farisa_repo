# MULTIMODAL DEEP LEARNING FOR AUTISM SPECTRUM DISORDER CLASSIFICATION USING STRUCTURAL AND FUNCTIONAL MRI

This project explores the use of multimodal deep learning to classify Autism Spectrum Disorder (ASD) from structural MRI (sMRI) and functional MRI (fMRI) data. 
Leveraging the Autism Brain Imaging Data Exchange (ABIDE I & II) dataset, the study integrates multiple neuroimaging modalities using 3D CNNs, BrainNetCNN, and Transformers, fused with an attention-based mechanism.

By combining complementary anatomical and functional brain features, the multimodal approach outperforms single-modality baselines and highlights the potential of deep learning for early ASD detection.

## Dataset

Source: https://fcon_1000.projects.nitrc.org/indi/abide/

Size: 1,025 subjects (488 ASD, 537 controls) across 18+ international sites

Modalities:

T1-weighted structural MRI

Resting-state fMRI (time series, 100 ROIs × 200 time points)

Functional connectivity matrices (100×100 Pearson correlations)

Preprocessing: fMRIPrep (skull stripping, motion correction, normalization, etc.)

