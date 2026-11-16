"""Types for planned attack jobs."""

from typing import Literal, TypeAlias, TypedDict

from ..configuration.schemas import Norm


InitializationMode: TypeAlias = Literal["image", "random"]


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
