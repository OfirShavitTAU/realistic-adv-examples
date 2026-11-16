"""Validate study inputs and display a plan. Does not load a model or run attacks."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


STUDY_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}


class PlanError(ValueError):
    """An invalid configuration or experiment group."""


################## VALIDATION HELPERS ##################

"""
    Checks a condition and stops planning if it is false.
    @type condition: bool
    @param condition: The condition required for valid input.
    @type message: str
    @param message: The error message for invalid input.
    @raises PlanError: If the condition is false.
"""
def require(condition, message):
    if not condition:
        raise PlanError(message)


"""
    Checks that a name can be used in the planned output folders.
    @type value: str
    @param value: The family, group, or image stem to validate.
    @rtype: str
    @returns: The validated name.
    @raises PlanError: If the name contains a path or invalid characters.
"""
def folder_name(value):
    valid_name = isinstance(value, str) and value not in {"", ".", ".."}
    require(valid_name, "Invalid folder name.")

    require(not any(character in value for character in '/\\:<>"|?*'), "Use folder names, not paths: " + value)
    require(not value.endswith((" ", ".")), "Invalid Windows folder name: " + value)

    return value


################## CONFIGURATION ##################

"""
    Reads the configuration and validates its groups and attack settings.
    @type path: Path
    @param path: The path to the JSON configuration file.
    @rtype: (dict, list)
    @returns: The configuration and a list of (family, group) pairs.
    @raises PlanError: If the configuration structure or settings are invalid.
"""
def read_config(path):
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)

    require(isinstance(config, dict), "Configuration must be a JSON object.")
    require(set(config) == {"groups", "random_baseline", "attack"},
            "Expected configuration keys: groups, random_baseline, attack.")

    selected = validate_groups(config["groups"])
    require(type(config["random_baseline"]) is bool, "random_baseline must be true or false.")
    validate_attack(config["attack"])

    return config, selected


"""
    Validates group names, family names, and image path lists.
    Image files themselves are checked later by read_images.
    @type groups: dict
    @param groups: The group definitions from the configuration.
    @rtype: list
    @returns: The (family, group) pairs in configuration order.
    @raises PlanError: If a group is invalid or its output path is repeated.
"""
def validate_groups(groups):
    require(isinstance(groups, dict) and groups, "groups must contain at least one group.")

    selected = []
    seen = set()

    for group, settings in groups.items():
        folder_name(group)
        require(isinstance(settings, dict) and set(settings) == {"family", "source", "initializations"},
                "Each group must contain family, source, and initializations: " + group)

        family = folder_name(settings["family"])
        key = (family.casefold(), group.casefold())
        require(key not in seen, "Repeated Windows group path: " + family + "/" + group)
        seen.add(key)

        for role in ("source", "initializations"):
            paths = settings[role]
            require(isinstance(paths, list) and paths and
                    all(isinstance(path, str) and path.strip() for path in paths),
                    group + "." + role + " must be a nonempty list of image paths.")

        selected.append((family, group))

    return selected


"""
    Checks the attack name, norms, seeds, and positive budgets.
    @type attack: dict
    @param attack: The attack settings from the configuration.
    @raises PlanError: If a required setting is missing or invalid.
"""
def validate_attack(attack):
    expected = {"name", "norms", "seeds", "max_queries", "max_unsafe_queries", "max_iterations"}
    require(isinstance(attack, dict) and set(attack) == expected, "Unexpected or missing attack settings.")
    require(attack["name"] == "hsja", "This study currently plans HSJA only.")

    norms = attack["norms"]
    valid_norms = (isinstance(norms, list) and norms and
                   all(norm in ("l2", "linf") for norm in norms))
    require(valid_norms, "norms must be a nonempty list containing l2 and/or linf.")
    require(len(set(norms)) == len(norms), "Duplicate norms.")

    seeds = attack["seeds"]
    valid_seeds = (isinstance(seeds, list) and seeds and
                   all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds))
    require(valid_seeds, "seeds must contain integers in [0, 2**32).")
    require(len(set(seeds)) == len(seeds), "Duplicate seeds.")

    for key in ("max_queries", "max_unsafe_queries", "max_iterations"):
        require(type(attack[key]) is int and attack[key] > 0, key + " must be a positive integer.")


################## IMAGES AND PATHS ##################

"""
    Resolves image paths and checks files, names, and image contents.
    @type paths: list
    @param paths: The configured image paths, in their original order.
    @type study_dir: Path
    @param study_dir: The base directory for relative image paths.
    @type context: str
    @param context: The group and role used in validation error messages.
    @rtype: list
    @returns: The absolute paths of the validated images.
    @raises PlanError: If images are missing, corrupt, or have conflicting stems.
"""
def read_images(paths, study_dir, context):
    images = []

    for value in paths:
        path = Path(value)

        if not path.is_absolute():
            path = study_dir / path

        path = path.resolve()
        require(path.is_file(), "Missing image in " + context + ": " + str(path))
        require(path.suffix.lower() in IMAGE_EXTENSIONS, "Unsupported image extension: " + str(path))
        images.append(path)

    # Results use image stems as directory/file names, so extensions cannot distinguish them.
    stems = [path.stem.casefold() for path in images]
    require(len(set(stems)) == len(stems), "Images with duplicate stems would overwrite results: " + context)

    from PIL import Image

    for path in images:
        folder_name(path.stem)
        require(path.stem.casefold() != "random", "The image stem 'random' is reserved for the baseline.")

        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as exc:
            raise PlanError("Cannot decode image: " + str(path)) from exc

    return images


"""
    Formats an image path for the plan.
    @type path: Path
    @param path: The resolved image path.
    @type study_dir: Path
    @param study_dir: The study directory used for relative paths.
    @rtype: str
    @returns: A relative path inside the study, otherwise an absolute path.
"""
def image_path(path, study_dir):
    try:
        return path.relative_to(study_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


################## EXPERIMENT PLAN ##################

"""
    Validates the images in all selected groups and builds the complete plan.
    @type config: dict
    @param config: The validated study configuration.
    @type selected: list
    @param selected: The (family, group) pairs to include.
    @type study_dir: Path
    @param study_dir: The base directory for image paths and output planning.
    @rtype: (list, list)
    @returns: The planned jobs and warnings about identical image contents.
    @raises PlanError: If any group contains invalid images.
"""
def build_plan(config, selected, study_dir=STUDY_DIR):
    jobs = []
    errors = []
    warnings = []

    # Validate each group before adding its jobs to the complete plan.
    for family, group in selected:
        settings = config["groups"][group]

        try:
            sources = read_images(settings["source"], study_dir, group + ".source")
            initializations = read_images(settings["initializations"], study_dir, group + ".initializations")
        except PlanError as exc:
            errors.append(str(exc))
            continue

        group_jobs, group_warnings = build_group_jobs(
            config, family, group, sources, initializations, study_dir
        )
        jobs.extend(group_jobs)
        warnings.extend(group_warnings)

    require(not errors, "Invalid experiment groups:\n  " + "\n  ".join(errors))

    return jobs, warnings


"""
    Pairs each source with all initializations in its own group.
    Adds one random baseline per source, norm, and seed when enabled.
    @type config: dict
    @param config: The study configuration containing attack and baseline settings.
    @type family: str
    @param family: The family used in output paths.
    @type group: str
    @param group: The group used in output paths.
    @type sources: list
    @param sources: The validated source image paths.
    @type initializations: list
    @param initializations: The validated initialization image paths.
    @type study_dir: Path
    @param study_dir: The base directory used to format image paths.
    @rtype: (list, list)
    @returns: The jobs for this group and warnings about identical images.
"""
def build_group_jobs(config, family, group, sources, initializations, study_dir):
    jobs = []
    warnings = []

    for source in sources:
        for initialization in initializations:
            if source.read_bytes() == initialization.read_bytes():
                warnings.append("Identical source and initialization: " + str(source) + " / " + str(initialization))

        # None keeps the original random initialization separate from image inputs.
        choices = [("image", path) for path in initializations]
        if config["random_baseline"]:
            choices.append(("random", None))

        for norm in config["attack"]["norms"]:
            for seed in config["attack"]["seeds"]:
                for mode, initialization in choices:
                    # Norm and seed folders prevent results from overwriting each other.
                    initialization_name = initialization.stem if initialization is not None else "random"
                    result = (Path("images") / family / group / "results" / source.stem /
                              (norm + "_seed_" + str(seed)) / (initialization_name + "-atttack.png"))

                    jobs.append({
                        "family": family,
                        "group": group,
                        "source": image_path(source, study_dir),
                        "initialization": image_path(initialization, study_dir)
                        if initialization is not None else None,
                        "initialization_mode": mode,
                        "norm": norm,
                        "seed": seed,
                        "result_image": result.as_posix(),
                    })

    return jobs, warnings


################## COMMAND LINE AND DISPLAY ##################

"""
    Reads command-line arguments, prepares the plan, and displays it.
    Does not load a model, run attacks, or create output directories.
    @type argv: list or None
    @param argv: Optional arguments; None uses the command-line arguments.
"""
def main(argv=None):
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


if __name__ == "__main__":
    main()
