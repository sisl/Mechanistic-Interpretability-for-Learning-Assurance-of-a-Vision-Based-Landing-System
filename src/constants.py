"""Fixed hyperparameters used across the paper."""

import torch

IMG_SIZE = 224
NUM_KEYPOINTS = 4  # LARDv2 runway corners: TL, TR, BL, BR

# ImageNet normalization for DINOv2 input.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
