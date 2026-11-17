"""Display study settings and planned jobs in the terminal."""

import json
from pathlib import Path

from ..configuration.schemas import SelectedGroup, StudyConfig
from .schemas import PlannedJob


################## PLAN DISPLAY ##################

"""
    Displays the settings, future output paths, and planned job counts.
    @type config: dict
    @param config: The validated study configuration.
    @type selected: list
    @param selected: The (family, group) pairs included in the plan.
    @type jobs: list
    @param jobs: The planned attack jobs.
    @type warnings: list
    @param warnings: The input warnings to display.
    @type config_path: Path
    @param config_path: The configuration file used for this plan.
    @type run_dir: Path
    @param run_dir: A preview of the future output directory; it is not created.
    @type plan_only: bool
    @param plan_only: Whether only planning was requested.
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
