"""Shared image processing matching the original ImageNet input pipeline."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


################## IMAGE PREPARATION ##################

"""
    Builds the spatial transforms used by both previews and model inputs.
    @rtype: torchvision.transforms.Compose
    @returns: Resize(256) followed by CenterCrop(224), as in src/dataset.py.
"""
def spatial_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])


"""
    Loads an image as a raw pixel tensor; normalization belongs to the model.
    @type path: Path
    @param path: The validated input image path.
    @rtype: torch.Tensor
    @returns: A float32 tensor of shape (1, 3, 224, 224) with values in [0, 1].
"""
def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        original = image.convert("RGB")

    processed = spatial_transforms()(original)
    tensor = transforms.ToTensor()(processed)

    return tensor.unsqueeze(0)
