# Projects
## 1. autism_prediction >
## Multimodal Deep Learning for Autism Spectrum Disorder Classification Using Structural and Functional MRI

This project focused on classifying Autism Spectrum Disorder (ASD) using the ABIDE I & II dataset containing T1-weighted structural MRI and resting-state fMRI. 
I built baseline models (3D CNN for sMRI, Transformer for fMRI time series, BrainNetCNN for connectivity) and a multimodal attention-based fusion model, which 
improved accuracy compared to single-modality models


## 2. mgexudb_association_score >
## Cumulative scoring of absolute expression calls associated with specific conditions and visualization
   

This project analyzed uterine gene expression data (MGEx-Udb) to compute Association Scores across conditions (disease, chemical treatment, chemoradiation, hormone treatment). 
I developed scripts to preprocess scores, normalize them, detect outliers, and generate scatter plots that highlight condition-specific biomarkers


## 3. predict_gene_expression >
## Machine Learning Pipeline for Single-Cell Perturbation Prediction

This project aimed to predict gene expression changes in human PBMCs after drug treatments using the Open Problems – Single-Cell Perturbations dataset
(a NeurIPS 2023 Kaggle competition). I enriched drug SMILES with ChemBERTa embeddings and implemented XGBoost, MLP, and GRU models for multi-output regression,
predicting how small molecules perturb gene expression across immune cell types
