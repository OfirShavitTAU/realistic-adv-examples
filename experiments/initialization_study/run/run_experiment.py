"""Run the initialization study on GPU by default; --plan-only stops before execution."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# Support direct Play in PyCharm as well as python -m execution.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

from experiments.initialization_study.progress import progress, stage
from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.integration.pipeline import run_pipeline
from experiments.initialization_study.planning.display import print_plan
from experiments.initialization_study.planning.planner import build_plan
from experiments.initialization_study.validation import PlanError


################## COMMAND LINE ##################

"""
    Builds the plan and runs the pipeline on GPU, or on CPU with --cpu.
    Use --plan-only to stop after planning. Execution saves each task; task errors cause a nonzero exit.
"""
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json",
                        help="Absolute path or path relative to the study directory (not the working directory).")

    mode_options = parser.add_mutually_exclusive_group()
    mode_options.add_argument("--plan-only", action="store_true",
                              help="Display the plan without running the model or saving results.")
    # Preserve existing launch configurations; execution is now the default.
    mode_options.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)

    device_options = parser.add_mutually_exclusive_group()
    device_options.add_argument("--device", default="cuda", help="CUDA GPU for execution, for example cuda:0.")
    device_options.add_argument("--cpu", action="store_true", help="Explicitly run on CPU instead of requiring CUDA.")

    args = parser.parse_args(argv)
    progress("Experiment entry point | config={} | mode={} | device={}".format(
        args.config, "plan only" if args.plan_only else "execute", "cpu" if args.cpu else args.device))
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = STUDY_DIR / config_path

    try:
        with stage('Read configuration'):
            config, selected = read_config(config_path)
        with stage('Build plan and validate images'):
            jobs, warnings = build_plan(config, selected)
        stamp = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d_%H-%M-%S-%f")
    except (PlanError, OSError, json.JSONDecodeError, ZoneInfoNotFoundError) as exc:
        parser.exit(1, "Planning failed: " + str(exc) + "\n")

    run_dir = STUDY_DIR / "results" / ("experiment-" + stamp)
    print_plan(config, selected, jobs, warnings, config_path, run_dir, plan_only=args.plan_only)

    if args.plan_only:
        return

    try:
        outcome = run_pipeline(config, selected, jobs, args.device, output_dir=run_dir, config_path=config_path, cpu=args.cpu)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, "Execution failed: " + str(error) + "\n")

    if any(task.status == "error" for task in outcome.tasks):
        parser.exit(1, "Execution finished with task errors; see the console output.\n")


if __name__ == "__main__":
    main()
