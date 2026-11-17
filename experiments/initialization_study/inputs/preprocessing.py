"""Shared image processing matching the original ImageNet input pipeline."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


################## IMAGE PREPARATION ##################

"""
    Creates Resize(256) and CenterCrop(224), matching src/dataset.py.
    Shared by the preview and model input preparation.
"""
def spatial_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])


"""
    Loads an RGB image as float32 pixels in [0, 1], shaped (1, 3, 224, 224).
    Applies the shared resize/crop; ImageNet normalization remains in the original model wrapper.
"""
def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        original = image.convert("RGB")

    processed = spatial_transforms()(original)
    tensor = transforms.ToTensor()(processed)

    return tensor.unsqueeze(0)
