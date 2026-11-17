"""Independent final-label verification on exact attack output pixels."""

from dataclasses import dataclass
from typing import Literal

import torch
from torchvision.models import ResNet50_Weights

from src.model_wrappers import TorchModelWrapper

from ..validation import require


@dataclass
class FinalValidation:
    """Final classification check; distance thresholds are not part of this success criterion."""

    status: Literal["checked", "error"]
    class_id: int | None = None
    class_name: str | None = None
    attack_success: bool | None = None
    queries: int = 0
    error: str | None = None


"""
    Classifies exact returned pixels without PNG rounding, clipping, or repeat resize/crop.
    Success means a final class different from the source's verified ground-truth ID.
    Keeps verification queries separate and reports failures without discarding the attack output.
"""
def verify_final_prediction(
    model: TorchModelWrapper, image: torch.Tensor, source_class: int, device: torch.device,
) -> FinalValidation:
    before = model.num_queries

    try:
        pixels = image.detach()

        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)

        require(tuple(pixels.shape) == (1, 3, 224, 224), "Invalid final image shape.")
        require(bool(torch.isfinite(pixels).all().item()), "Final image contains nonfinite pixels.")
        require(bool(((pixels >= 0) & (pixels <= 1)).all().item()), "Final image pixels are outside [0, 1].")

        with torch.no_grad():
            class_id = int(model.predict_label(pixels.to(device)).item())

        categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        require(0 <= class_id < len(categories), "Invalid final ImageNet class ID.")

        return FinalValidation("checked", class_id, categories[class_id], class_id != source_class,
                               model.num_queries - before)
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        return FinalValidation("error", queries=model.num_queries - before,
                               error=type(error).__name__ + ": " + str(error))
