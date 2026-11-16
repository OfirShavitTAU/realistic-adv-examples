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


def require(condition, message):
    if not condition:
        raise PlanError(message)


def folder_name(value):
    require(isinstance(value, str) and value not in {"", ".", ".."}, "Invalid folder name.")
    require(not any(c in value for c in '/\\:<>"|?*'), "Use folder names, not paths: " + value)
    require(not value.endswith((" ", ".")), "Invalid Windows folder name: " + value)
    return value


def read_config(path):
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    require(isinstance(config, dict), "Configuration must be a JSON object.")
    require(set(config) == {"experiment_groups", "random_baseline", "attack"},
            "Expected configuration keys: experiment_groups, random_baseline, attack.")
    groups = config["experiment_groups"]
    require(isinstance(groups, dict) and groups, "experiment_groups must contain families and groups.")
    selected = []
    seen = set()
    for family, group_flags in groups.items():
        folder_name(family)
        require(isinstance(group_flags, dict) and group_flags, "Each family must contain group names and booleans.")
        for group, enabled in group_flags.items():
            folder_name(group)
            require(type(enabled) is bool, "Group selections must be true or false.")
            key = (family.casefold(), group.casefold())
            require(key not in seen, "Repeated Windows group path: " + family + "/" + group)
            seen.add(key)
            if enabled:
                selected.append((family, group))
    require(selected, "Select at least one group with true.")
    require(type(config["random_baseline"]) is bool, "random_baseline must be true or false.")
    attack = config["attack"]
    expected = {"name", "norms", "seeds", "max_queries", "max_unsafe_queries", "max_iterations"}
    require(isinstance(attack, dict) and set(attack) == expected, "Unexpected or missing attack settings.")
    require(attack["name"] == "hsja", "This study currently plans HSJA only.")
    norms = attack["norms"]
    require(isinstance(norms, list) and norms and all(n in ("l2", "linf") for n in norms),
            "norms must be a nonempty list containing l2 and/or linf.")
    require(len(set(norms)) == len(norms), "Duplicate norms.")
    seeds = attack["seeds"]
    require(isinstance(seeds, list) and seeds and
            all(type(s) is int and 0 <= s < 2**32 for s in seeds), "seeds must contain integers in [0, 2**32).")
    require(len(set(seeds)) == len(seeds), "Duplicate seeds.")
    for key in ("max_queries", "max_unsafe_queries", "max_iterations"):
        require(type(attack[key]) is int and attack[key] > 0, key + " must be a positive integer.")
    return config, selected


def read_images(directory):
    require(directory.is_dir(), "Missing directory: " + str(directory))
    require(not any(p.is_dir() for p in directory.iterdir()),
            "Place image files directly inside: " + str(directory))
    images = sorted((p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: p.name.casefold())
    require(images, "No images in: " + str(directory))
    # Results use image stems as directory/file names, so extensions cannot distinguish them.
    stems = [p.stem.casefold() for p in images]
    require(len(set(stems)) == len(stems), "Images with duplicate stems would overwrite results: " + str(directory))
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


def build_plan(config, selected, study_dir=STUDY_DIR):
    jobs = []
    errors = []
    warnings = []
    for family, group in selected:
        group_dir = study_dir / "experiment_groups" / family / group
        try:
            sources = read_images(group_dir / "source")
            initializations = read_images(group_dir / "initializations")
        except PlanError as exc:
            errors.append(str(exc))
            continue
        for source in sources:
            for initialization in initializations:
                if source.read_bytes() == initialization.read_bytes():
                    warnings.append("Identical source and initialization: " + str(source) + " / " + str(initialization))
            choices = [("image", p) for p in initializations]
            if config["random_baseline"]:
                choices.append(("random", None))
            for norm in config["attack"]["norms"]:
                for seed in config["attack"]["seeds"]:
                    for mode, initialization in choices:
                        init_name = initialization.stem if initialization is not None else "random"
                        result = (Path("images") / family / group / "results" / source.stem /
                                  (norm + "_seed_" + str(seed)) / (init_name + "-atttack.png"))
                        jobs.append({
                            "family": family,
                            "group": group,
                            "source": source.relative_to(study_dir).as_posix(),
                            "initialization": initialization.relative_to(study_dir).as_posix()
                            if initialization is not None else None,
                            "initialization_mode": mode,
                            "norm": norm,
                            "seed": seed,
                            "result_image": result.as_posix(),
                        })
    require(not errors, "Invalid experiment groups:\n  " + "\n  ".join(errors))
    return jobs, warnings


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
    print("PLAN ONLY - no model, attacks, or output directories are created.")
    print("Config: " + str(config_path.resolve()))
    print("Future output: " + str(run_dir))
    print("Model for execution stage: torchvision ResNet-50 / IMAGENET1K_V1")
    print("Attack settings: " + json.dumps(config["attack"]))
    print("Selected groups: " + ", ".join(f + "/" + g for f, g in selected))
    for warning in warnings:
        print("WARNING: " + warning)
    for i, job in enumerate(jobs, 1):
        start = job["initialization"] or "random noise (original HSJA initialization)"
        print("\n{}. {} / {} | {} | seed {}".format(i, job["family"], job["group"], job["norm"], job["seed"]))
        print("  Source: " + job["source"])
        print("  Initialization: " + start)
        print("  Result: " + job["result_image"])
    random_count = sum(j["initialization_mode"] == "random" for j in jobs)
    print("\nTotal: {} attacks ({} image initializations + {} random baselines).".format(
        len(jobs), len(jobs) - random_count, random_count))
    print("Model predictions, source labels, and initialization validity will be checked at execution time.")


if __name__ == "__main__":
    main()
