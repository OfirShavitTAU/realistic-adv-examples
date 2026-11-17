"""Orchestrate model preparation checks, original HSJA tasks, and incremental output storage."""

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
from ..outputs.writer import RunWriter
from ..run.run_preflight import check_pairs, check_sources, collect_groups, predict_images
from .hsja import AttackOutput, run_hsja
from .model import load_target_model, select_device
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
    source_tensor: torch.Tensor | None = None
    initialization_tensor: torch.Tensor | None = None


@dataclass
class PipelineResult:
    """Preparation predictions, outcomes, and the directory containing their saved artifacts."""

    preparation_queries: int
    predictions: dict[Path, ImagePrediction]
    tasks: list[TaskResult]
    sources: list[SourceCheck]
    output_dir: Path | None = None


################## TASK PREPARATION ##################

"""
    Seeds each task; CUDA-specific generators are used only in GPU mode.
"""
def set_task_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        # Seed the CPU generator directly, without invoking CUDA through torch.manual_seed.
        torch.random.default_generator.manual_seed(seed)


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
    source_pixels = None
    initialization_pixels = None

    try:
        source_pixels = load_image_tensor(source_path)
        source = source_pixels.to(device)
        initialization = None

        if initialization_path is not None:
            initialization_pixels = load_image_tensor(initialization_path)
            initialization = initialization_pixels.to(device)

        source_label = torch.tensor([source_class], dtype=torch.long, device=device)
        set_task_seed(job["seed"], device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        output = run_hsja(model, source, source_label, config["attack"], job["norm"], initialization)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Keep exact pixels in CPU memory for immediate artifact storage.
        image, counter, distance, success, extra = output
        cpu_output: AttackOutput = (image.detach().cpu(), counter, distance, success, extra)

        return TaskResult(job, "completed", perf_counter() - started, model.num_queries, cpu_output,
                          source_true_class_id=source_class, source_tensor=source_pixels,
                          initialization_tensor=initialization_pixels)
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        return TaskResult(job, "error", perf_counter() - started, model.num_queries,
                          reason=type(error).__name__ + ": " + str(error), source_true_class_id=source_class,
                          source_tensor=source_pixels, initialization_tensor=initialization_pixels)


################## PIPELINE ##################

"""
    Loads one model on CUDA or explicitly on CPU and saves each task incrementally.
    Preparation, running tasks, errors, and interruptions are recorded separately.
    Final-result verification is still pending; reported HSJA success is preserved as reported.
"""
def run_pipeline(
    config: StudyConfig, selected: list[SelectedGroup], jobs: list[PlannedJob],
    device_name: str = "cuda", output_dir: Path | None = None, config_path: Path | None = None,
    cpu: bool = False,
) -> PipelineResult:
    device = select_device(device_name, cpu=cpu)
    groups = collect_groups(config, selected)
    print("Execution device: " + str(device))
    writer = RunWriter(config, jobs, str(device), output_dir, config_path)
    model: TorchModelWrapper | None = None
    preparation_complete = False
    tasks: list[TaskResult] = []
    print("Output directory: " + str(writer.directory))

    try:
        writer.copy_inputs(selected, groups)
        model = load_target_model(str(device), cpu=cpu)
        predictions = predict_images(model, groups)
        source_checks = check_sources(selected, groups, predictions)
        pair_checks = check_pairs(selected, groups, predictions)
        preparation_queries = model.num_queries
        writer.save_preflight(predictions, source_checks, pair_checks, preparation_queries)
        preparation_complete = True

        print("\nSource predictions checked against ImageNet labels inferred from class folders.")
        print("Each task is saved immediately; independent final-result verification remains not_checked.")

        # Fixed seeds are repeatable on a fixed setup, not a cross-device guarantee.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        for index, job in enumerate(jobs):
            print("\nTask {}/{}: {} / {} | {} | seed {} | {}".format(
                index + 1, len(jobs), job["family"], job["group"], job["norm"], job["seed"],
                job["initialization"] or "random baseline",
            ))
            writer.start_task(index)
            result = run_task(model, config, job, predictions, device)
            writer.save_task(index, result)
            tasks.append(result)
            print("{} | model queries: {} | task time: {:.3f}s".format(
                result.status, result.model_queries, result.elapsed_seconds,
            ))

            if result.reason is not None:
                print(result.reason)

        status = "completed_with_errors" if any(task.status == "error" for task in tasks) else "completed"
        writer.finish(status)
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        partial_preparation = model.num_queries if model is not None and not preparation_complete else None

        try:
            writer.finish(status, type(error).__name__ + ": " + str(error), partial_preparation)
        except OSError as storage_error:
            print("Unable to update run status: " + str(storage_error))

        raise

    print("\nPreparation queries: {} | Actual attack model queries: {}".format(
        preparation_queries, sum(task.model_queries for task in tasks),
    ))
    print("Completed: {} | Skipped: {} | Errors: {}".format(
        sum(task.status == "completed" for task in tasks),
        sum(task.status == "skipped" for task in tasks),
        sum(task.status == "error" for task in tasks),
    ))
    print("Saved results: " + str(writer.directory))

    return PipelineResult(preparation_queries, predictions, tasks, source_checks, writer.directory)
