import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized

def generate_heatmap():
    ROOT_DIR = '/home/martou/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

    # Get first valid patient
    df = pd.read_csv(os.path.join(ROOT_DIR, 'name_mapping.csv')).dropna(subset=['Grade', 'BraTS_2020_subject_ID'])
    patient = df.iloc[0]['BraTS_2020_subject_ID']

    subj_dir = os.path.join(ROOT_DIR, patient)
    data = {
        "flair": os.path.join(subj_dir, f"{patient}_flair.nii"),
        "t1": os.path.join(subj_dir, f"{patient}_t1.nii"),
        "t1ce": os.path.join(subj_dir, f"{patient}_t1ce.nii"),
        "t2": os.path.join(subj_dir, f"{patient}_t2.nii")
    }

    transforms = Compose([
        LoadImaged(keys=["flair", "t1", "t1ce", "t2"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ScaleIntensityd(keys=["flair", "t1", "t1ce", "t2"]),
        Resized(keys=["flair", "t1", "t1ce", "t2"], spatial_size=(96, 96, 96))
    ])

    print("[*] Loading and transforming 3D volumes for correlation analysis...")
    vol = transforms(data)
    
    # Flatten the 3D volumes into 1D arrays for pixel-wise correlation
    flair_flat = vol["flair"].numpy().flatten()
    t1_flat = vol["t1"].numpy().flatten()
    t1ce_flat = vol["t1ce"].numpy().flatten()
    t2_flat = vol["t2"].numpy().flatten()

    # Filter out background pixels (where all are roughly 0) to get true tissue correlation
    # We create a mask where at least one modality has intensity > 0.05
    mask = (flair_flat > 0.05) | (t1_flat > 0.05) | (t1ce_flat > 0.05) | (t2_flat > 0.05)

    df_corr = pd.DataFrame({
        'FLAIR': flair_flat[mask],
        'T1': t1_flat[mask],
        'T1ce': t1ce_flat[mask],
        'T2': t2_flat[mask]
    })

    print("[*] Computing Pearson Correlation Matrix...")
    corr_matrix = df_corr.corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='magma', vmin=0, vmax=1, fmt=".3f", linewidths=1)
    plt.title(f'Спектрална Корелация между ЯМР Модалностите\n(Patient: {patient})', fontweight='bold', fontsize=12)
    
    output_file = 'brats_correlation_heatmap.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"[*] Saved correlation heatmap to {output_file}")

if __name__ == "__main__":
    generate_heatmap()
