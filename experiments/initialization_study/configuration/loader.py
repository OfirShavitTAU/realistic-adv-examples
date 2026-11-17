"""Read and validate initialization study settings."""

import json
from pathlib import Path
from typing import Literal

from .schemas import AttackConfig, GroupConfig, ImageRole, SelectedGroup, StudyConfig
from ..validation import folder_name, require


STUDY_DIR = Path(__file__).resolve().parents[1]


################## CONFIGURATION ##################

"""
    Reads and validates the JSON configuration.
    Returns the settings and selected (family, group) pairs in configuration order.
"""
def read_config(path: Path) -> tuple[StudyConfig, list[SelectedGroup]]:
    with path.open(encoding="utf-8-sig") as stream:
        config: StudyConfig = json.load(stream)

    require(isinstance(config, dict), "Configuration must be a JSON object.")
    require(set(config) == {"groups", "random_baseline", "attack"},
            "Expected configuration keys: groups, random_baseline, attack.")

    selected = validate_groups(config["groups"])
    require(type(config["random_baseline"]) is bool, "random_baseline must be true or false.")
    validate_attack(config["attack"])

    return config, selected


"""
    Checks group definitions, image path lists, and output-name conflicts.
    Validates source class-folder labels here; image files are checked separately.
"""
def validate_groups(groups: dict[str, GroupConfig]) -> list[SelectedGroup]:
    require(isinstance(groups, dict) and groups, "groups must contain at least one group.")

    selected: list[SelectedGroup] = []
    seen: set[SelectedGroup] = set()

    for group, settings in groups.items():
        folder_name(group)
        require(isinstance(settings, dict) and set(settings) == {"family", "source", "initializations"},
                "Each group must contain family, source, and initializations: " + group)

        family = folder_name(settings["family"])
        key = (family.casefold(), group.casefold())
        require(key not in seen, "Repeated Windows group path: " + family + "/" + group)
        seen.add(key)

        roles: tuple[ImageRole, ImageRole] = ("source", "initializations")

        for role in roles:
            paths = settings[role]
            require(isinstance(paths, list) and paths and
                    all(isinstance(path, str) and path.strip() for path in paths),
                    group + "." + role + " must be a nonempty list of image paths.")

        # Class IDs belong to the image organization, not the user-facing configuration.
        for image_path in settings["source"]:
            get_source_label(Path(image_path))

        selected.append((family, group))

    return selected


"""
    Checks the HSJA name, supported norms, unique seeds, and positive budgets.
"""
def validate_attack(attack: AttackConfig) -> None:
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

    budget_keys: tuple[Literal["max_queries", "max_unsafe_queries", "max_iterations"], ...] = (
        "max_queries", "max_unsafe_queries", "max_iterations"
    )

    for key in budget_keys:
        require(type(attack[key]) is int and attack[key] > 0, key + " must be a positive integer.")


"""
    Reads the ImageNet ID from a class folder such as 207_golden_retriever.
    Uses the immediate parent of the resolved image path; invalid names or IDs raise PlanError.
"""
def get_source_label(source_path: Path) -> int:
    resolved_source = (STUDY_DIR / source_path).resolve()
    class_folder = resolved_source.parent.name
    prefix, separator, class_name = class_folder.partition("_")

    valid_folder = (separator == "_" and prefix.isascii() and prefix.isdecimal()
                    and bool(class_name.strip()))
    require(valid_folder,
            "Cannot infer source label. Place the image in an <ImageNet ID>_<class name> folder "
            "such as 207_golden_retriever: " + str(resolved_source))

    label = int(prefix)
    require(0 <= label < 1000, "ImageNet class folder ID must be in [0, 999]: " + class_folder)

    return label
