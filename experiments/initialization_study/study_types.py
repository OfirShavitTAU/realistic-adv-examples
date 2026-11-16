"""Shared types for validated configuration, planned jobs, and preview inputs."""

from pathlib import Path
from typing import Literal, TypeAlias, TypedDict


################## CONFIGURATION ##################

Norm: TypeAlias = Literal["l2", "linf"]
InitializationMode: TypeAlias = Literal["image", "random"]
ImageRole: TypeAlias = Literal["source", "initializations"]
SelectedGroup: TypeAlias = tuple[str, str]
PreviewEntry: TypeAlias = tuple[str, str, ImageRole, Path]


class GroupConfig(TypedDict):
    """Image selections and family for one experiment group."""

    family: str
    source: list[str]
    initializations: list[str]


class AttackConfig(TypedDict):
    """Validated attack settings; budgets are used by the future execution stage."""

    name: Literal["hsja"]
    norms: list[Norm]
    seeds: list[int]
    max_queries: int
    max_unsafe_queries: int
    max_iterations: int


class StudyConfig(TypedDict):
    """The complete validated study configuration."""

    groups: dict[str, GroupConfig]
    random_baseline: bool
    attack: AttackConfig


################## PLANNED JOBS ##################

class PlannedJob(TypedDict):
    """One planned source/initialization combination and its future output path."""

    family: str
    group: str
    source: str
    initialization: str | None
    initialization_mode: InitializationMode
    norm: Norm
    seed: int
    result_image: str
