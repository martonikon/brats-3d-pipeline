import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from data_loader import get_brats_dataloaders
from model import BraTSClassifier

def evaluate_model():
    """
    Evaluates the 3D-CNN on the validation dataset.
    Computes classification metrics and saves ROC Curve and Confusion Matrix plots.
    """
    ROOT_DIR = '/home/martou/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    MODEL_PATH = 'best_brats_3d_classifier.pth'
    BATCH_SIZE = 2 # Keep it small for memory
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing evaluation on: {device}")
    
    # 1. Load Data
    # The split in data_loader is sequential, so the validation set is always the exact same 20%
    _, val_loader = get_brats_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE)
    print(f"[*] Validation set size: {len(val_loader.dataset)} patients")

    # 2. Load Model
    model = BraTSClassifier().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[*] Loaded trained weights from '{MODEL_PATH}'")
    else:
        print(f"[!] Critical: Model weights '{MODEL_PATH}' not found. Evaluation will fail.")
        return
        
    model.eval()

    # 3. Evaluation Loop
    print("[*] Running inference on validation data. This might take a minute...")
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(dtype=torch.float32)
            
            outputs = model(inputs).squeeze(-1) # shape: [Batch] or empty if batch is 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            
            if (i + 1) % 5 == 0:
                print(f"    - Processed batch {i + 1}/{len(val_loader)}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_preds_binary = (all_preds >= 0.5).astype(int)

    # 4. Metrics Calculation
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT (HGG vs LGG)")
    print("="*40)
    # target_names: 0 = LGG, 1 = HGG
    print(classification_report(all_labels, all_preds_binary, target_names=['LGG', 'HGG']))
    
    # 5. Plotting Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
    plt.title("Confusion Matrix (Validation Set)")
    plt.ylabel('True Diagnosis')
    plt.xlabel('AI Predicted Diagnosis')
    
    cm_path = "brats_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved Confusion Matrix to: {cm_path}")

    # 6. Plotting ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = "brats_roc_curve.png"
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved ROC Curve to: {roc_path}")
    print("[*] Evaluation complete!")

if __name__ == "__main__":
    evaluate_model()
