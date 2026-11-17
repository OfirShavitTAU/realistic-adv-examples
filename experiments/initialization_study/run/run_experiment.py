"""Plan the initialization study, or explicitly execute its GPU-only HSJA pipeline."""

import argparse
import json
import sys
from importlib import import_module
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# Support direct Play in PyCharm as well as python -m execution.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.planning.display import print_plan
from experiments.initialization_study.planning.planner import build_plan
from experiments.initialization_study.validation import PlanError


################## COMMAND LINE ##################

"""
    Prepares the plan and optionally executes it when --execute is supplied.
    The default remains plan-only; result persistence is not implemented yet.
    @type argv: list or None
    @param argv: Optional arguments; None uses the command-line arguments.
"""
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json",
                        help="Absolute path or path relative to the study directory (not the working directory).")

    parser.add_argument("--execute", action="store_true",
                        help="Run GPU-only preflight and HSJA tasks. Results are not yet saved to disk.")
    parser.add_argument("--device", default="cuda", help="CUDA GPU for execution, for example cuda:0.")

    args = parser.parse_args(argv)
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = STUDY_DIR / config_path

    try:
        config, selected = read_config(config_path)
        jobs, warnings = build_plan(config, selected)
        stamp = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d_%H-%M-%S-%f")
    except (PlanError, OSError, json.JSONDecodeError, ZoneInfoNotFoundError) as exc:
        parser.exit(1, "Planning failed: " + str(exc) + "\n")

    run_dir = STUDY_DIR / "results" / ("experiment-" + stamp)
    print_plan(config, selected, jobs, warnings, config_path, run_dir, plan_only=not args.execute)

    if not args.execute:
        return

    # Import execution dependencies only for explicit execution, keeping planning lightweight.
    pipeline = import_module("experiments.initialization_study.integration.pipeline")

    try:
        outcome = pipeline.run_pipeline(config, selected, jobs, args.device)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, "Execution failed: " + str(error) + "\n")

    if any(task.status == "error" for task in outcome.tasks):
        parser.exit(1, "Execution finished with task errors; see the console output.\n")


if __name__ == "__main__":
    main()
