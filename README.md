# 3D BraTS Diagnostic Pipeline

This repository contains a full Deep Learning pipeline for the classification of High-Grade Gliomas (HGG) vs Low-Grade Gliomas (LGG) using the MICCAI BraTS 2020 dataset.

## Architecture
The diagnostic model utilizes a **3D DenseNet-121** architecture via the [MONAI](https://project-monai.github.io/) framework. It processes 4-channel Multi-modal MRI volumes (T1, T1ce, T2, FLAIR) which are digitally fused and reduced to a $96 \times 96 \times 96$ tensor.

## Pipeline Components
- `data_loader.py`: Handles loading, fusion, downsampling, and spatial 3D augmentations (shuffling, rotating, flipping).
- `model.py`: Neural network configuration (`spatial_dims=3`).
- `train_brats.py`: Complete training loop with Early Stopping, L2 Regularization, and `ReduceLROnPlateau`.
- `evaluate_brats.py`: Generates the Confusion Matrix and ROC Curve metrics over the validation set.
- `test_brats.py`: Performs visual inference, overlaying the ground-truth tumor masks onto the corresponding slice.
- `generate_heatmap.py`: Generates a Pearson correlation heatmap across the 4 MRI modalities.

## How to use
1. Install dependencies: `pip install -r requirements.txt`
2. Download BraTS 2020 dataset and adjust the `ROOT_DIR` in the scripts.
3. Run `train_brats.py` to initiate the training loop.
