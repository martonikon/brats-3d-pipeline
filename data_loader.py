import os
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, 
    Resized, ScaleIntensityd, EnsureTyped
)

def get_brats_dataloaders(root_dir, batch_size=2):
    """
    Constructs PyTorch dataloaders for the BraTS 2020 dataset using MONAI.
    Fuses 4 MRI modalities (FLAIR, T1, T1ce, T2) into a single 4-channel 3D volume.
    """
    csv_path = os.path.join(root_dir, 'name_mapping.csv')
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid BraTS_2020_subject_ID and Grade
    df = df.dropna(subset=['Grade', 'BraTS_2020_subject_ID'])
    
    # Map HGG to 1 (High Grade) and LGG to 0 (Low Grade)
    grade_map = {'HGG': 1, 'LGG': 0}
    
    data_dicts = []
    
    print(f"[*] Parsing metadata and constructing image paths from: {root_dir}")
    for _, row in df.iterrows():
        subject_id = row['BraTS_2020_subject_ID']
        grade = grade_map.get(row['Grade'], None)
        
        if grade is None:
            continue
            
        subj_dir = os.path.join(root_dir, subject_id)
        if not os.path.exists(subj_dir):
            continue
            
        # Define paths to all 4 modalities
        flair = os.path.join(subj_dir, f"{subject_id}_flair.nii")
        t1 = os.path.join(subj_dir, f"{subject_id}_t1.nii")
        t1ce = os.path.join(subj_dir, f"{subject_id}_t1ce.nii")
        t2 = os.path.join(subj_dir, f"{subject_id}_t2.nii")
        
        if os.path.exists(flair) and os.path.exists(t1) and os.path.exists(t1ce) and os.path.exists(t2):
            data_dicts.append({
                "flair": flair,
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
                "label": grade
            })

    print(f"[*] Valid records found: {len(data_dicts)}")

    from monai.transforms import RandRotate90d, RandFlipd
    
    # -------------------------------------------------------------
    # 3D Data Augmentation & Preprocessing Pipeline (Fusion)
    # -------------------------------------------------------------
    train_transforms = Compose([
        LoadImaged(keys=["flair", "t1", "t1ce", "t2"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        # Random augmentations for robust training
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image", "label"])
    ])
    
    # Validation MUST NOT have random augmentations
    val_transforms = Compose([
        LoadImaged(keys=["flair", "t1", "t1ce", "t2"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image", "label"])
    ])
    
    import random
    # Fix: SHUFFLE the dataset before splitting! 
    # The original CSV has all HGGs first and LGGs last. Without shuffling, 
    # the training set becomes ~85% HGG, causing the model to always guess HGG.
    random.seed(42) # Set seed for reproducible splits
    random.shuffle(data_dicts)
    
    # Split the dataset 80% / 20%
    train_size = int(0.8 * len(data_dicts))
    train_files = data_dicts[:train_size]
    val_files = data_dicts[train_size:]
    
    # Calculate and print class distribution to ensure balance
    train_hgg = sum(1 for d in train_files if d['label'] == 1)
    train_lgg = sum(1 for d in train_files if d['label'] == 0)
    val_hgg = sum(1 for d in val_files if d['label'] == 1)
    val_lgg = sum(1 for d in val_files if d['label'] == 0)
    
    print(f"[*] Train set: {len(train_files)} (HGG: {train_hgg}, LGG: {train_lgg})")
    print(f"[*] Val set: {len(val_files)} (HGG: {val_hgg}, LGG: {val_lgg})")
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
