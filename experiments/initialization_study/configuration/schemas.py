"""Types for the study configuration and selected groups."""

from typing import Literal, TypeAlias, TypedDict


Norm: TypeAlias = Literal["l2", "linf"]
ImageRole: TypeAlias = Literal["source", "initializations"]
SelectedGroup: TypeAlias = tuple[str, str]


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
