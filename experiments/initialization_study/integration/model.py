"""Load the researchers' original model wrapper on a CUDA GPU."""

import torch
from torchvision.models import ResNet50_Weights, resnet50

from src.model_wrappers import TorchModelWrapper

from ..validation import PlanError


################## DEVICE SELECTION ##################

"""
    Selects an available CUDA GPU; CPU fallback is disabled.
    Invalid or unavailable devices raise PlanError before any model weights are loaded.
"""
def require_cuda(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise PlanError("Invalid device: " + device_name) from error

    if device.type != "cuda":
        raise PlanError("Model inference requires a CUDA GPU. CPU execution is disabled.")

    if not torch.cuda.is_available():
        raise PlanError("CUDA is unavailable. Use the GPU VM; there is no CPU fallback.")

    device_index = device.index if device.index is not None else torch.cuda.current_device()

    if device_index >= torch.cuda.device_count():
        raise PlanError("CUDA device does not exist: " + device_name)

    return torch.device("cuda", device_index)


################## TARGET MODEL ##################

"""
    Loads ResNet-50 V1 on CUDA and returns the original TorchModelWrapper.
    The original wrapper handles normalization, predictions, and query counting.
    Torchvision downloads weights on this call if they are not already cached.
"""
def load_target_model(device_name: str = "cuda") -> TorchModelWrapper:
    device = require_cuda(device_name)
    inner_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

    # Their wrapper uses .cuda() for normalization tensors, so select this GPU explicitly.
    with torch.cuda.device(device):
        model = TorchModelWrapper(
            inner_model,
            n_class=1000,
            im_mean=(0.485, 0.456, 0.406),
            im_std=(0.229, 0.224, 0.225),
        )

    model.make_model_eval()

    return model
