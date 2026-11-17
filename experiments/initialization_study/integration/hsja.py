"""Minimal device-aware adapter to the researchers' original untargeted HSJA."""

from typing import TypeAlias

import torch
from foolbox.distances import l2, linf

from src.attacks.base import Bounds, ExtraResultsDict
from src.attacks.hsja import HSJA, HSJAttackPhase
from src.attacks.queries_counter import QueriesCounter
from src.model_wrappers import TorchModelWrapper

from ..configuration.schemas import AttackConfig, Norm
from ..validation import require


AttackOutput: TypeAlias = tuple[
    torch.Tensor, QueriesCounter[HSJAttackPhase], float, bool, ExtraResultsDict
]


################## ATTACK SETUP ##################

"""
    Creates the original HSJA with the configured budgets and norm.
    Other algorithm parameters keep the researchers' defaults.
"""
def create_hsja(settings: AttackConfig, norm: Norm) -> HSJA:
    require(settings["name"] == "hsja", "Only HSJA is supported.")
    require(norm in settings["norms"], "The task norm must be enabled in the configuration.")
    require(norm in ("l2", "linf"), "HSJA requires l2 or linf.")

    return HSJA(
        epsilon=None,
        distance=l2 if norm == "l2" else linf,
        bounds=Bounds(0, 1),
        discrete=False,
        queries_limit=settings["max_queries"],
        unsafe_queries_limit=settings["max_unsafe_queries"],
        num_iterations=settings["max_iterations"],
    )


################## INPUT CONTRACT ##################

"""
    Checks single-image shapes, dtypes, and placement on the model's selected CPU or CUDA device.
    Images must contain raw [0, 1] pixels shaped (1, 3, 224, 224); labels use shape (1,).
    Source correctness and initialization predictions must be checked by the caller.
"""
def validate_inputs(
    model: TorchModelWrapper, source: torch.Tensor, source_label: torch.Tensor,
    initialization: torch.Tensor | None,
) -> None:
    require(source.device.type in ("cpu", "cuda"), "HSJA supports CPU or CUDA tensors only.")
    require(tuple(source.shape) == (1, 3, 224, 224), "HSJA expects exactly one source: (1, 3, 224, 224).")
    require(source.dtype == torch.float32, "HSJA source must use float32 raw pixels.")
    require(next(model.parameters()).device == source.device, "Model and source must be on the same device.")

    require(tuple(source_label.shape) == (1,), "HSJA expects one source label with shape (1,).")
    require(source_label.dtype == torch.long, "The source label must use torch.long.")
    require(source_label.device == source.device, "Source label and image must be on the same device.")

    if initialization is not None:
        require(initialization.device == source.device, "Initialization and source must be on the same device.")
        require(initialization.shape == source.shape, "Initialization and source must have the same shape.")
        require(initialization.dtype == source.dtype, "Initialization and source must use the same dtype.")


################## ORIGINAL ATTACK CALL ##################

"""
    Calls the original untargeted HSJA with an image initialization or None for original random noise.
    Returns its (image, counter, distance, success, extra) tuple unchanged.
    The success flag needs independent verification; original query budgets are not strict limits.
"""
def run_hsja(
    model: TorchModelWrapper, source: torch.Tensor, source_label: torch.Tensor,
    settings: AttackConfig, norm: Norm, initialization: torch.Tensor | None = None,
) -> AttackOutput:
    validate_inputs(model, source, source_label, initialization)
    attack = create_hsja(settings, norm)

    # HSJA removes the source batch dimension internally, but not the initialization's.
    target_image = initialization[0] if initialization is not None else None

    # Match HSJA.__call__ parameters, changing only the optional initialization image.
    with torch.no_grad():
        return attack.hsja(
            model=model,
            sample=source,
            original_label=source_label,
            clip_max=attack.bounds.upper,
            clip_min=attack.bounds.lower,
            distance=attack.distance,
            num_iterations=attack.num_iterations,
            gamma=attack.gamma,
            fixed_delta=attack.fixed_delta,
            target_label=None,
            target_image=target_image,
            stepsize_search=attack.stepsize_search,
            max_num_evals=attack.max_num_evals,
            init_num_evals=attack.init_num_evals,
            bias_coef=attack.bias_coef,
            lower_bad_query_bound=attack.lower_bad_query_bound,
            upper_bad_query_bound=attack.upper_bad_query_bound,
            bias_coef_change_rate=attack.bias_coef_change_rate,
        )
