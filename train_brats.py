import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_brats_dataloaders
from model import BraTSClassifier

def train():
    # 1. Configuration
    # Make sure to point this to the correct extracted dataset path
    ROOT_DIR = '/home/martou/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    BATCH_SIZE = 2 # 3D CNNs are memory heavy. Keep this at 2 (or 1) for most GPUs
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    EARLY_STOP_PATIENCE = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing 3D training on: {device}")

    # 2. Data Loaders
    print("\n[*] Preparing fused 3D data pipeline...")
    train_loader, val_loader = get_brats_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE)

    # 3. Model & Optimizer
    print("\n[*] Initializing 3D-CNN (DenseNet121)...")
    model = BraTSClassifier().to(device)
    
    # Binary Cross Entropy Loss for Binary Classification (HGG vs LGG)
    criterion = nn.BCELoss()
    # Added L2 regularization (weight_decay) to combat overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # Reduce learning rate when learning stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 4. Training Loop
    print("\n[*] Starting 3D Fusion Classification Training...")
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # MONAI dataloader returns a dictionary
            inputs = batch["image"].to(device)
            # Labels need to be float and shaped [Batch, 1]
            labels = batch["label"].to(device, dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate error
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"===> End of Epoch {epoch+1}. Mean Loss: {epoch_loss:.4f}")
        
        # Step the scheduler
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Current Learning Rate: {current_lr}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_brats_3d_classifier.pth')
            print(f"[*] Best model weights saved! (Loss: {best_loss:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[*] EARLY STOPPING triggered after {epoch+1} epochs! The model has reached its peak.")
                break

if __name__ == "__main__":
    train()
