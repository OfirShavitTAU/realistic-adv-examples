"""Display study settings and planned jobs in the terminal."""

import json


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
"""
def print_plan(config, selected, jobs, warnings, config_path, run_dir):
    print("PLAN ONLY - no model, attacks, or output directories are created.")
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
