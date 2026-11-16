"""Build source/initialization jobs without executing attacks."""

from pathlib import Path

from .configuration import STUDY_DIR
from .images import image_path, read_images
from .validation import PlanError, require


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
