"""Types for model predictions and preparation checks."""

from typing import Literal, TypeAlias, TypedDict


SourceLabelStatus: TypeAlias = Literal["correct", "incorrect"]


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
    source_true_class_id: int
    source_label_status: SourceLabelStatus
    eligible: bool


class SourceCheck(TypedDict):
    """A source prediction compared with the ground-truth ID inferred from its class folder."""

    family: str
    group: str
    source: str
    true_class_id: int
    predicted_class_id: int
    source_label_status: SourceLabelStatus
