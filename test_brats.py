import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, 
    Resized, ScaleIntensityd, EnsureTyped
)
from model import BraTSClassifier

def test_and_visualize():
    """
    Evaluates the trained 3D-CNN on a random validation sample 
    and generates a qualitative visual output (2D slice) 
    with the Ground Truth tumor mask overlaid in RED.
    """
    # 1. Configuration
    ROOT_DIR = '/home/martou/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    MODEL_PATH = 'best_brats_3d_classifier.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing inference pipeline on: {device}")
    
    # 2. Load Model
    model = BraTSClassifier().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[*] Successfully loaded trained weights from '{MODEL_PATH}'")
    else:
        print(f"[!] Warning: '{MODEL_PATH}' not found. Using untrained (randomized) weights.")
        
    model.eval() # Freeze layers for inference

    # 3. Select a Random Patient
    csv_path = os.path.join(ROOT_DIR, 'name_mapping.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Grade', 'BraTS_2020_subject_ID'])
    
    random_patient = df.sample(n=1).iloc[0]
    subject_id = random_patient['BraTS_2020_subject_ID']
    true_grade = random_patient['Grade']
    
    print(f"\n[*] Processing Patient: {subject_id} (Ground Truth: {true_grade})")
    
    # 4. Construct Paths (including SEGMENTATION mask)
    subj_dir = os.path.join(ROOT_DIR, subject_id)
    test_data = {
        "flair": os.path.join(subj_dir, f"{subject_id}_flair.nii"),
        "t1": os.path.join(subj_dir, f"{subject_id}_t1.nii"),
        "t1ce": os.path.join(subj_dir, f"{subject_id}_t1ce.nii"),
        "t2": os.path.join(subj_dir, f"{subject_id}_t2.nii"),
        "seg": os.path.join(subj_dir, f"{subject_id}_seg.nii") # <-- Ground Truth Mask
    }
    
    # 5. Apply Fusion Transformations
    # Note: we also resize the 'seg' so it matches the 96x96x96 dimensions perfectly
    # mode="nearest" is crucial for segmentation masks to avoid interpolating class labels
    val_transforms = Compose([
        LoadImaged(keys=["flair", "t1", "t1ce", "t2", "seg"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2", "seg"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "seg"], spatial_size=(96, 96, 96), mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "seg"])
    ])
    
    transformed_data = val_transforms(test_data)
    # Add batch dimension to image -> Shape: [1, 4, 96, 96, 96]
    input_tensor = transformed_data["image"].unsqueeze(0).to(device)
    
    # Extract the segmentation mask -> Shape: [1, 96, 96, 96]
    seg_tensor = transformed_data["seg"]
    
    # 6. Inference Execution
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        
    pred_grade = "HGG" if prob >= 0.5 else "LGG"
    print(f"[*] AI Prediction Confidence (Probability of HGG): {prob:.4f}")
    print(f"[*] AI Final Diagnosis: {pred_grade}")
    
    # 7. 2D Slice Visualization from 3D Volume
    seg_numpy = seg_tensor[0].cpu().numpy() # [96, 96, 96]
    image_numpy = input_tensor[0].cpu().numpy() # [4, 96, 96, 96]
    flair_volume = image_numpy[0]
    
    # Find the depth (Z) index with the maximum number of tumor pixels (>0)
    # This guarantees we look at the most obvious part of the tumor
    tumor_pixels_per_slice = [np.sum(seg_numpy[:, :, z] > 0) for z in range(96)]
    best_z = np.argmax(tumor_pixels_per_slice)
    
    if tumor_pixels_per_slice[best_z] == 0:
        best_z = 48 # Fallback if no tumor found
        
    best_flair_slice = flair_volume[:, :, best_z]
    best_seg_slice = seg_numpy[:, :, best_z]
    
    # Rotate 90 degrees so the brain is upright
    best_flair_slice = np.rot90(best_flair_slice)
    best_seg_slice = np.rot90(best_seg_slice)
    
    plt.figure(figsize=(10, 10))
    # Plot the base FLAIR slice
    plt.imshow(best_flair_slice, cmap='gray')
    
    # Overlay the tumor mask in RED where mask > 0
    # We use np.ma.masked_where to make the 0 (background) transparent
    masked_tumor = np.ma.masked_where(best_seg_slice == 0, best_seg_slice)
    plt.imshow(masked_tumor, cmap='hsv', alpha=0.5) 
    
    color = 'lime' if pred_grade == true_grade else 'red'
    plt.title(f"Patient: {subject_id} | Slice Depth: {best_z}/96\n"
              f"Truth: {true_grade} | AI Predicted: {pred_grade} ({prob:.2%})", 
              color=color, fontweight='bold', fontsize=14)
    plt.axis('off')
    
    # Add a custom legend to explain the overlay
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.5, label='Tumor Region (Mask)')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    output_img = f"brats_test_result_{subject_id}.png"
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n[*] Sliced visualization persisted to: {output_img}")

if __name__ == "__main__":
    test_and_visualize()
