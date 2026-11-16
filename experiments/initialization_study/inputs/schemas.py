"""Types for image preview inputs."""

from pathlib import Path
from typing import TypeAlias

from ..configuration.schemas import ImageRole


PreviewEntry: TypeAlias = tuple[str, str, ImageRole, Path]
