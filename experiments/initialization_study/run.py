"""Validate study inputs and display a plan. Does not load a model or run attacks."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .configuration.loader import STUDY_DIR, read_config
from .planning.display import print_plan
from .planning.planner import build_plan
from .validation import PlanError


################## COMMAND LINE ##################

"""
    Reads command-line arguments, prepares the plan, and displays it.
    Does not load a model, run attacks, or create output directories.
    @type argv: list or None
    @param argv: Optional arguments; None uses the command-line arguments.
"""
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json",
                        help="Absolute path or path relative to the study directory (not the working directory).")

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
    print_plan(config, selected, jobs, warnings, config_path, run_dir)


if __name__ == "__main__":
    main()
