"""Minimal GPU-only adapter to the researchers' original untargeted HSJA."""

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
    Creates a fresh original HSJA instance using the validated study settings.
    All remaining algorithm parameters keep the original constructor defaults.
    @type settings: AttackConfig
    @param settings: Validated attack budgets and iteration count.
    @type norm: Norm
    @param norm: The norm for this individual task.
    @rtype: HSJA
    @returns: The researchers' attack object, not a replacement implementation.
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
    Checks the tensor shapes and GPU placement required by the original attack.
    Source correctness and initialization predictions must be checked by the caller.
    @type model: TorchModelWrapper
    @param model: The original model wrapper loaded on a CUDA GPU.
    @type source: torch.Tensor
    @param source: Raw float32 pixels in [0, 1], shaped (1, 3, 224, 224).
    @type source_label: torch.Tensor
    @param source_label: A single class ID, shaped (1,), on the same GPU.
    @type initialization: torch.Tensor or None
    @param initialization: Raw pixels with the source shape, or None for random noise.
"""
def validate_inputs(
    model: TorchModelWrapper, source: torch.Tensor, source_label: torch.Tensor,
    initialization: torch.Tensor | None,
) -> None:
    require(source.device.type == "cuda", "HSJA source must be on a CUDA GPU; CPU is disabled.")
    require(tuple(source.shape) == (1, 3, 224, 224), "HSJA expects exactly one source: (1, 3, 224, 224).")
    require(source.dtype == torch.float32, "HSJA source must use float32 raw pixels.")
    require(next(model.parameters()).device == source.device, "Model and source must be on the same CUDA GPU.")

    require(tuple(source_label.shape) == (1,), "HSJA expects one source label with shape (1,).")
    require(source_label.dtype == torch.long, "The source label must use torch.long.")
    require(source_label.device == source.device, "Source label and image must be on the same GPU.")

    if initialization is not None:
        require(initialization.device == source.device, "Initialization and source must be on the same GPU.")
        require(initialization.shape == source.shape, "Initialization and source must have the same shape.")
        require(initialization.dtype == source.dtype, "Initialization and source must use the same dtype.")


################## ORIGINAL ATTACK CALL ##################

"""
    Calls the original HSJA with an external image or its original random baseline.
    No optimization, initialization search, query accounting, or stopping is rewritten.
    @type model: TorchModelWrapper
    @param model: The original GPU model wrapper.
    @type source: torch.Tensor
    @param source: Prepared raw source pixels; ImageNet normalization stays in the model.
    @type source_label: torch.Tensor
    @param source_label: The source class against which classification change is tested.
    @type settings: AttackConfig
    @param settings: Validated study attack settings.
    @type norm: Norm
    @param norm: The norm for this individual task.
    @type initialization: torch.Tensor or None
    @param initialization: Prepared image, or None to use the researchers' random initialization.
    @rtype: AttackOutput
    @returns: The original (image, counter, distance, success flag, extra results) tuple.
        The success flag is not an independently verified study outcome.
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
