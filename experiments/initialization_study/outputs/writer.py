"""Incremental experiment output using the agreed image tree and exact tensor artifacts."""

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import torch
from PIL import Image

from ..configuration.loader import STUDY_DIR
from ..configuration.schemas import SelectedGroup, StudyConfig
from ..integration.hsja import create_hsja
from ..integration.schemas import ImagePrediction, PairCheck, SourceCheck
from ..planning.schemas import PlannedJob
from ..validation import require
from .serialization import write_csv, write_json, write_tensor

if TYPE_CHECKING:
    from ..integration.pipeline import TaskResult


COLUMNS = [
    "task_id", "family", "group", "source", "initialization", "initialization_mode", "norm", "seed",
    "status", "reason", "source_true_class_id", "source_predicted_class_id", "source_predicted_class_name",
    "source_label_status",
    "initialization_predicted_class_id", "initialization_predicted_class_name", "elapsed_seconds", "actual_model_queries",
    "reported_total_queries", "reported_unsafe_queries", "reported_safe_queries",
    "attack_reported_distance", "attack_reported_success", "attack_outcome", "attack_success", "final_validation_status",
    "final_predicted_class_id", "final_predicted_class_name", "validation_error",
    "l2", "linf", "distance_validation_status", "validation_queries", "total_task_model_queries",
    "result_image", "result_tensor", "task_metadata", "query_history",
]


"""
    Returns an Israel timestamp suitable for chronologically sorted run directories.
"""
def israel_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d_%H-%M-%S-%f")


"""
    Records local versions and Git state without loading a model or accessing the network.
"""
def environment_info(device_name: str) -> dict[str, object]:
    packages: dict[str, str | None] = {}

    for package in ("torch", "torchvision", "numpy", "Pillow", "foolbox", "torch-dct"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    git_info: dict[str, object] = {"commit": None, "dirty": None}

    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=STUDY_DIR,
                                  capture_output=True, text=True, check=True, timeout=5)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=STUDY_DIR,
                                capture_output=True, text=True, check=True, timeout=5)
        git_info = {"commit": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass

    gpu_name = None

    if torch.device(device_name).type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device(device_name))

    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "git": git_info, "device": device_name, "gpu_name": gpu_name,
        "torch_cuda_version": torch.version.cuda,
    }


"""
    Writes one display PNG without changing the exact tensor saved alongside it.
    Rounding and clipping apply only to this 8-bit visualization.
"""
def write_image(path: Path, tensor: torch.Tensor) -> None:
    pixels = tensor.detach().cpu()

    if pixels.ndim == 4 and pixels.shape[0] == 1:
        pixels = pixels[0]

    require(tuple(pixels.shape) == (3, 224, 224), "Expected a (3, 224, 224) output image.")
    display = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(display).save(temporary, format="PNG")
    temporary.replace(path)


class RunWriter:
    """One new run directory; summary rows and run state are updated after every task."""

    """
        Creates the run snapshot and pending summary rows before model loading.
        Refuses existing directories so previous runs cannot be overwritten.
    """
    def __init__(
        self, config: StudyConfig, jobs: list[PlannedJob], device_name: str,
        output_dir: Path | None = None, config_path: Path | None = None,
    ) -> None:
        destinations = [job["result_image"].casefold() for job in jobs]
        require(len(set(destinations)) == len(destinations), "Planned output image paths must be unique.")
        self.directory = output_dir if output_dir is not None else STUDY_DIR / "results" / ("experiment-" + israel_timestamp())
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.jobs = jobs
        self.rows: list[dict[str, object]] = []
        self.predictions: dict[Path, ImagePrediction] = {}

        for index, job in enumerate(jobs):
            row: dict[str, object] = {column: None for column in COLUMNS}
            row.update({key: value for key, value in job.items() if key in COLUMNS})
            row.update({"task_id": self.task_id(index), "status": "pending", "final_validation_status": "not_checked",
                        "actual_model_queries": 0, "validation_queries": 0, "result_image": None,
                        "attack_outcome": "pending", "distance_validation_status": "not_checked", "total_task_model_queries": 0})
            self.rows.append(row)

        effective_parameters: dict[str, object] = {}

        for norm in config["attack"]["norms"]:
            attack = create_hsja(config["attack"], norm)
            effective_parameters[norm] = {**vars(attack), "distance": norm, "bounds": attack.bounds._asdict()}

        self.info: dict[str, object] = {
            "status": "preparing", "started_at": datetime.now(ZoneInfo("Asia/Jerusalem")).isoformat(),
            "finished_at": None, "config_path": str(config_path.resolve()) if config_path else None,
            "environment": environment_info(device_name), "model": "resnet50", "weights": "IMAGENET1K_V1",
            "preprocessing": ["RGB", "Resize(256)", "CenterCrop(224)", "ToTensor", "ImageNet normalization"],
            "effective_attack_parameters": effective_parameters, "total_tasks": len(jobs), "finished_tasks": 0,
            "active_task_id": None, "preparation_queries": 0, "actual_attack_model_queries": 0,
            "validation_queries": 0, "final_validation_status": "not_checked", "source_label_origin": "class_directory",
            "distance_validation_status": "not_checked",
            "success_definition": "final_prediction_differs_from_verified_source_ground_truth",
        }
        write_json(self.directory / "config.json", config)
        write_json(self.directory / "data/plan.json", jobs)
        self.flush()

    """
        Returns a stable artifact ID independent of image names, norms, and seeds.
    """
    def task_id(self, index: int) -> str:
        return "task-{:06d}".format(index + 1)

    """
        Copies original files into the agreed family/group/role tree and records their hashes.
        Sources are preserved byte-for-byte rather than re-encoded.
    """
    def copy_inputs(
        self, selected: list[SelectedGroup], groups: dict[str, tuple[list[Path], list[Path]]],
    ) -> None:
        manifest: list[dict[str, object]] = []

        for family, group in selected:
            sources, initializations = groups[group]

            for role, paths in (("source", sources), ("initializations", initializations)):
                for source in paths:
                    relative = Path("images") / family / group / role / source.name
                    copied = self.directory / relative
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, copied)
                    digest = hashlib.sha256()

                    with copied.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)

                    manifest.append({"family": family, "group": group, "role": role,
                                     "original_path": str(source), "copied_path": relative.as_posix(),
                                     "sha256": digest.hexdigest()})

        write_json(self.directory / "data/input_manifest.json", manifest)

    """
        Saves cached source/pair checks and separates preparation cost from attack cost.
    """
    def save_preflight(
        self, predictions: dict[Path, ImagePrediction], sources: list[SourceCheck],
        pairs: list[PairCheck], preparation_queries: int,
    ) -> None:
        self.predictions = predictions
        write_json(self.directory / "data/preflight.json", {
            "predictions": list(predictions.values()), "sources": sources, "pairs": pairs,
            "preparation_queries": preparation_queries,
        })
        self.info.update({"preparation_queries": preparation_queries, "status": "running"})
        self.flush()

    """
        Marks a task running before invoking HSJA, making interruptions visible on disk.
    """
    def start_task(self, index: int) -> None:
        self.rows[index]["status"] = "running"
        self.rows[index]["attack_outcome"] = "running"
        self.info["active_task_id"] = self.task_id(index)
        self.flush()

    """
        Stores task artifacts first, then publishes the completed/skipped/error summary row.
        Reported HSJA success remains separate from verified classification success and final class.
    """
    def save_task(self, index: int, result: "TaskResult") -> None:
        require(result.job == self.jobs[index], "Task result does not match the planned task.")
        require(result.status != "completed" or result.output is not None,
                "A completed task must include the original attack output.")
        row = dict(self.rows[index])
        job = result.job
        data_dir = Path("data") / self.task_id(index)
        source = (STUDY_DIR / job["source"]).resolve()
        initialization = (STUDY_DIR / job["initialization"]).resolve() if job["initialization"] else None
        row.update({"status": result.status, "reason": result.reason, "source_label_status": result.source_label_status,
                    "source_true_class_id": result.source_true_class_id, "elapsed_seconds": result.elapsed_seconds,
                    "actual_model_queries": result.model_queries,
                    "source_predicted_class_id": self.predictions[source]["class_id"],
                    "source_predicted_class_name": self.predictions[source]["class_name"],
                    "attack_outcome": result.outcome, "attack_success": result.attack_success,
                    "final_validation_status": result.final_validation_status,
                    "final_predicted_class_id": result.final_predicted_class_id,
                    "final_predicted_class_name": result.final_predicted_class_name,
                    "validation_queries": result.validation_queries, "validation_error": result.validation_error,
                    "total_task_model_queries": result.model_queries + result.validation_queries})

        if initialization is not None:
            row["initialization_predicted_class_id"] = self.predictions[initialization]["class_id"]
            row["initialization_predicted_class_name"] = self.predictions[initialization]["class_name"]

        for name, tensor in (("source", result.source_tensor), ("initialization", result.initialization_tensor)):
            if tensor is not None:
                write_tensor(self.directory / data_dir / (name + ".pt"), tensor)

        history_status = "unavailable"
        query_details: dict[str, object] | None = None
        extra: object = None

        if result.output is not None:
            image, counter, distance, success, extra = result.output
            output_tensor = data_dir / "output.pt"
            write_tensor(self.directory / output_tensor, image)
            write_image(self.directory / job["result_image"], image)
            history_path = data_dir / "queries.json"
            query_details = {
                "reported_total_queries": counter.total_queries,
                "reported_unsafe_queries": counter.total_unsafe_queries,
                "queries_by_phase": counter.queries, "unsafe_queries_by_phase": counter.unsafe_queries,
                "history": counter.distances,
            }
            write_json(self.directory / history_path, query_details)
            row.update({"reported_total_queries": counter.total_queries,
                        "reported_unsafe_queries": counter.total_unsafe_queries,
                        "reported_safe_queries": counter.total_queries - counter.total_unsafe_queries,
                        "attack_reported_distance": distance, "attack_reported_success": success,
                        "result_image": job["result_image"], "result_tensor": output_tensor.as_posix(),
                        "query_history": history_path.as_posix()})
            history_status = "recorded"

        metadata_path = data_dir / "task.json"
        row["task_metadata"] = metadata_path.as_posix()
        write_json(self.directory / metadata_path, {
            "job": job, "summary": row, "extra_results": extra,
            "query_history_status": history_status,
            "reported_query_counts": ({key: value for key, value in query_details.items() if key != "history"}
                                      if query_details is not None else None),
            "final_validation_status": result.final_validation_status,
            "distance_validation_status": "not_checked",
        })
        self.rows[index] = row
        self.info["finished_tasks"] = sum(item["status"] in ("completed", "skipped", "error") for item in self.rows)
        self.info["actual_attack_model_queries"] = sum(int(item["actual_model_queries"] or 0) for item in self.rows)
        self.info["active_task_id"] = None
        self.flush()

    """
        Atomically updates the CSV and run metadata after each observable state change.
    """
    def flush(self) -> None:
        outcome_names = ("success", "failure", "skipped", "error", "not_checked", "pending", "running")
        task_ids = {name: [row["task_id"] for row in self.rows if row["attack_outcome"] == name]
                    for name in outcome_names}
        counts = {name: len(identifiers) for name, identifiers in task_ids.items()}
        verified = counts["success"] + counts["failure"]
        self.info["outcomes"] = counts
        self.info["task_ids_by_outcome"] = task_ids
        self.info["verified_tasks"] = verified
        self.info["verified_success_rate"] = counts["success"] / verified if verified else None
        self.info["validation_queries"] = sum(int(row["validation_queries"] or 0) for row in self.rows)
        self.info["total_model_queries"] = (int(self.info["preparation_queries"]) +
                                           int(self.info["actual_attack_model_queries"]) +
                                           int(self.info["validation_queries"]))
        terminal_rows = [row for row in self.rows if row["status"] in ("completed", "skipped", "error")]
        validation_errors = any(row["final_validation_status"] == "error" for row in self.rows)

        if validation_errors:
            self.info["final_validation_status"] = "partial" if verified else "error"
        elif verified:
            all_checked = (len(terminal_rows) == len(self.rows) and not counts["not_checked"]
                           and not counts["error"])
            self.info["final_validation_status"] = "checked" if all_checked else "partial"
        else:
            self.info["final_validation_status"] = "not_checked"

        write_csv(self.directory / "summary.csv", self.rows, COLUMNS)
        write_json(self.directory / "run_info.json", self.info)

    """
        Records normal completion, a setup/storage failure, or an explicit interruption.
        Earlier task artifacts and summary rows are kept intact.
    """
    def finish(
        self, status: str, error: str | None = None, preparation_queries: int | None = None,
    ) -> None:
        if preparation_queries is not None:
            self.info["preparation_queries"] = preparation_queries

        self.info.update({"status": status, "error": error,
                          "finished_at": datetime.now(ZoneInfo("Asia/Jerusalem")).isoformat()})
        self.flush()
