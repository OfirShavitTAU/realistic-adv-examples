"""Build source/initialization jobs without executing attacks."""

from pathlib import Path

from ..configuration.loader import STUDY_DIR
from ..inputs.images import image_path, read_images
from ..configuration.schemas import SelectedGroup, StudyConfig
from .schemas import InitializationMode, PlannedJob
from ..validation import PlanError, require


################## EXPERIMENT PLAN ##################

"""
    Validates group images and combines their planned tasks and warnings.
    Raises PlanError if any group is invalid; no partial plan is returned.
"""
def build_plan(
    config: StudyConfig, selected: list[SelectedGroup], study_dir: Path = STUDY_DIR
) -> tuple[list[PlannedJob], list[str]]:
    jobs: list[PlannedJob] = []
    errors: list[str] = []
    warnings: list[str] = []

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
    Pairs each source with every initialization in its own group for each norm and seed.
    Adds a random baseline when enabled and warns about byte-identical source/initialization files.
"""
def build_group_jobs(
    config: StudyConfig, family: str, group: str, sources: list[Path],
    initializations: list[Path], study_dir: Path
) -> tuple[list[PlannedJob], list[str]]:
    jobs: list[PlannedJob] = []
    warnings: list[str] = []

    for source in sources:
        for initialization in initializations:
            if source.read_bytes() == initialization.read_bytes():
                warnings.append("Identical source and initialization: " + str(source) + " / " + str(initialization))

        # None selects the researchers' original random initialization.
        choices: list[tuple[InitializationMode, Path | None]] = [
            ("image", path) for path in initializations
        ]
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
