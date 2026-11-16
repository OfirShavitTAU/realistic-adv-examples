"""Types for model predictions and preparation checks."""

from typing import Literal, TypedDict


################## PREFLIGHT REPORTS ##################

class ImagePrediction(TypedDict):
    """A target-model prediction for one unique configured image."""

    path: str
    class_id: int
    class_name: str


class PairCheck(TypedDict):
    """Classification-change check, separate from source ground-truth correctness."""

    family: str
    group: str
    source: str
    initialization: str
    source_class_id: int
    initialization_class_id: int
    changes_source_prediction: bool
    source_label_status: Literal["not_checked"]
