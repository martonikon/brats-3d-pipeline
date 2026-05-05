import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

class BraTSClassifier(nn.Module):
    """
    3D Convolutional Neural Network (3D-CNN) for binary classification
    of brain tumors (HGG vs LGG) using a fused 4-channel input.
    """
    def __init__(self):
        super(BraTSClassifier, self).__init__()
        
        # We utilize MONAI's 3D implementation of DenseNet121
        # spatial_dims=3 makes it a 3D CNN (using Conv3d instead of Conv2d)
        # in_channels=4 because we fused 4 MRI modalities (FLAIR, T1, T1ce, T2)
        # out_channels=1 because this is binary classification (0 or 1)
        self.densenet = DenseNet121(
            spatial_dims=3,
            in_channels=4,
            out_channels=1
        )
        
        # Sigmoid to convert the raw output into a probability [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is expected to have shape: [Batch_Size, 4, Depth, Height, Width]
        out = self.densenet(x)
        return self.sigmoid(out)

if __name__ == "__main__":
    # Test the model structure with a dummy 3D tensor
    model = BraTSClassifier()
    # 1 batch, 4 channels (modalities), 96x96x96 spatial dimensions
    dummy_input = torch.randn(1, 4, 96, 96, 96)
    
    output = model(dummy_input)
    print(f"[*] Model successfully initialized!")
    print(f"[*] Input shape: {dummy_input.shape}")
    print(f"[*] Output shape: {output.shape} (Probability: {output.item():.4f})")
