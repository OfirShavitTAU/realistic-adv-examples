"""Orchestrate GPU preparation checks and original HSJA tasks without persistence."""

import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch

from src.model_wrappers import TorchModelWrapper

from ..configuration.loader import STUDY_DIR, get_source_label
from ..configuration.schemas import SelectedGroup, StudyConfig
from ..inputs.preprocessing import load_image_tensor
from ..planning.schemas import PlannedJob
from ..run.run_preflight import check_pairs, check_sources, collect_groups, predict_images
from .hsja import AttackOutput, run_hsja
from .model import load_target_model, require_cuda
from .schemas import ImagePrediction, SourceCheck, SourceLabelStatus


################## EXECUTION RESULTS ##################

@dataclass
class TaskResult:
    """One task outcome; completed means returned, not independently verified success."""

    job: PlannedJob
    status: Literal["completed", "skipped", "error"]
    elapsed_seconds: float = 0.0
    model_queries: int = 0
    output: AttackOutput | None = None
    reason: str | None = None
    source_label_status: SourceLabelStatus = "correct"
    source_true_class_id: int | None = None


@dataclass
class PipelineResult:
    """In-memory preparation predictions and task outcomes; no files are written."""

    preparation_queries: int
    predictions: dict[Path, ImagePrediction]
    tasks: list[TaskResult]
    sources: list[SourceCheck]


################## TASK PREPARATION ##################

"""
    Seeds Python, NumPy, PyTorch, and CUDA for one planned task.
"""
def set_task_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
    Resolves a planned image path against the study directory.
    Uses the same absolute keys as the cached preparation predictions.
"""
def resolve_image(value: str) -> Path:
    return (STUDY_DIR / value).resolve()


################## SINGLE TASK ##################

"""
    Checks source and initialization eligibility, seeds the task, and calls the HSJA adapter.
    Skips misclassified sources and same-class initializations without attack queries.
    Returns the outcome and actual query count; task errors are captured and exact output pixels move to CPU.
"""
def run_task(
    model: TorchModelWrapper, config: StudyConfig, job: PlannedJob,
    predictions: dict[Path, ImagePrediction], device: torch.device,
) -> TaskResult:
    source_path = resolve_image(job["source"])
    source_class = get_source_label(source_path)

    if predictions[source_path]["class_id"] != source_class:
        return TaskResult(
            job, "skipped", reason="Source prediction does not match its ground-truth label.",
            source_label_status="incorrect", source_true_class_id=source_class,
        )

    initialization_path = resolve_image(job["initialization"]) if job["initialization"] is not None else None

    if initialization_path is not None:
        initialization_class = predictions[initialization_path]["class_id"]

        if initialization_class == source_class:
            return TaskResult(job, "skipped", reason="Initialization has the same prediction as the source.",
                              source_true_class_id=source_class)

    # Keep each task's actual query count separate from preparation.
    model.num_queries = 0
    started = perf_counter()

    try:
        source = load_image_tensor(source_path).to(device)
        initialization = None

        if initialization_path is not None:
            initialization = load_image_tensor(initialization_path).to(device)

        source_label = torch.tensor([source_class], dtype=torch.long, device=device)
        set_task_seed(job["seed"])
        torch.cuda.synchronize(device)

        output = run_hsja(model, source, source_label, config["attack"], job["norm"], initialization)
        torch.cuda.synchronize(device)

        # Preserve exact pixels in CPU memory; disk storage is still pending.
        image, counter, distance, success, extra = output
        cpu_output: AttackOutput = (image.detach().cpu(), counter, distance, success, extra)

        return TaskResult(job, "completed", perf_counter() - started, model.num_queries, cpu_output,
                          source_true_class_id=source_class)
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        return TaskResult(job, "error", perf_counter() - started, model.num_queries,
                          reason=type(error).__name__ + ": " + str(error), source_true_class_id=source_class)


################## PIPELINE ##################

"""
    Loads one CUDA model, runs preparation checks, and executes the planned tasks in order.
    Separates preparation queries from attack queries and continues after recorded task errors.
    Returns results in memory only; disk persistence and final-result verification are still pending.
"""
def run_pipeline(
    config: StudyConfig, selected: list[SelectedGroup], jobs: list[PlannedJob],
    device_name: str = "cuda",
) -> PipelineResult:
    device = require_cuda(device_name)
    groups = collect_groups(config, selected)
    model = load_target_model(str(device))

    predictions = predict_images(model, groups)
    source_checks = check_sources(selected, groups, predictions)
    check_pairs(selected, groups, predictions)
    preparation_queries = model.num_queries

    print("\nSource predictions checked against ImageNet labels inferred from class folders.")
    print("Results are currently returned in memory only; output persistence and verification are pending.")
    tasks: list[TaskResult] = []

    # Fixed seeds are repeatable on a fixed software/device setup, not a cross-device guarantee.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for task_number, job in enumerate(jobs, 1):
        print("\nTask {}/{}: {} / {} | {} | seed {} | {}".format(
            task_number, len(jobs), job["family"], job["group"], job["norm"], job["seed"],
            job["initialization"] or "random baseline",
        ))
        result = run_task(model, config, job, predictions, device)
        tasks.append(result)
        print("{} | model queries: {} | task time: {:.3f}s".format(
            result.status, result.model_queries, result.elapsed_seconds,
        ))

        if result.reason is not None:
            print(result.reason)

    print("\nPreparation queries: {} | Actual attack model queries: {}".format(
        preparation_queries, sum(task.model_queries for task in tasks),
    ))
    print("Completed: {} | Skipped: {} | Errors: {}".format(
        sum(task.status == "completed" for task in tasks),
        sum(task.status == "skipped" for task in tasks),
        sum(task.status == "error" for task in tasks),
    ))

    return PipelineResult(preparation_queries, predictions, tasks, source_checks)
