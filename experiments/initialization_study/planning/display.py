"""Display study settings and planned jobs in the terminal."""

import json
from pathlib import Path

from ..configuration.schemas import SelectedGroup, StudyConfig
from .schemas import PlannedJob


################## PLAN DISPLAY ##################

"""
    Displays the configuration, planned tasks, warnings, and future output paths.
    The output directory is only a preview and is not created here.
"""
def print_plan(
    config: StudyConfig, selected: list[SelectedGroup], jobs: list[PlannedJob],
    warnings: list[str], config_path: Path, run_dir: Path, plan_only: bool = True
) -> None:
    if plan_only:
        print("PLAN ONLY - no model, attacks, or output directories are created.")
    else:
        print("EXECUTION REQUESTED - GPU-only HSJA; disk output persistence is not implemented yet.")

    print("Config: " + str(config_path.resolve()))
    print("Future output: " + str(run_dir))
    print("Model for execution stage: torchvision ResNet-50 / IMAGENET1K_V1")
    print("Attack settings: " + json.dumps(config["attack"]))
    print("Selected groups: " + ", ".join(family + "/" + group for family, group in selected))

    for warning in warnings:
        print("WARNING: " + warning)

    for job_number, job in enumerate(jobs, 1):
        start = job["initialization"] or "random noise (original HSJA initialization)"
        print("\n{}. {} / {} | {} | seed {}".format(job_number, job["family"], job["group"], job["norm"], job["seed"]))
        print("  Source: " + job["source"])
        print("  Initialization: " + start)
        print("  Result: " + job["result_image"])

    random_count = sum(job["initialization_mode"] == "random" for job in jobs)
    print("\nTotal: {} attacks ({} image initializations + {} random baselines).".format(
        len(jobs), len(jobs) - random_count, random_count))
    print("Model predictions, source labels, and initialization validity will be checked at execution time.")
