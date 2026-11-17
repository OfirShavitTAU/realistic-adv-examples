"""Load the original model wrapper on CUDA by default, or explicitly on CPU."""

import torch
from torchvision.models import ResNet50_Weights, resnet50

from src.model_wrappers import TorchModelWrapper

from ..progress import stage
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
        raise PlanError("GPU mode requires a CUDA device. Select --cpu explicitly to run on CPU.")

    if not torch.cuda.is_available():
        raise PlanError("CUDA is unavailable. Select --cpu explicitly to run on CPU; there is no automatic fallback.")

    device_index = device.index if device.index is not None else torch.cuda.current_device()

    if device_index >= torch.cuda.device_count():
        raise PlanError("CUDA device does not exist: " + device_name)

    return torch.device("cuda", device_index)


"""
    Uses CPU only when explicitly requested; otherwise requires an available CUDA GPU.
    Device validation happens before model creation or weight downloads.
"""
def select_device(device_name: str = "cuda", cpu: bool = False) -> torch.device:
    if cpu:
        return torch.device("cpu")

    return require_cuda(device_name)


################## TARGET MODEL ##################

"""
    Loads ResNet-50 V1 on the selected device and returns the original TorchModelWrapper.
    Keeps the original normalization, predictions, and query counting without modifying src.
    Torchvision downloads weights on this call if they are not already cached.
"""
def load_target_model(device_name: str = "cuda", cpu: bool = False) -> TorchModelWrapper:
    device = select_device(device_name, cpu=cpu)
    with stage("Construct ResNet-50 and load weights (may download on first use)"):
        inner_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    with stage("Transfer ResNet-50 to " + str(device)):
        inner_model = inner_model.to(device).eval()

    # The original constructor hardcodes .cuda() for supplied statistics.
    # Assign the same statistics afterward on the selected device, preserving its preprocessing code.
    model = TorchModelWrapper(inner_model, n_class=1000)
    model.im_mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    model.im_std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    model.make_model_eval()

    return model
