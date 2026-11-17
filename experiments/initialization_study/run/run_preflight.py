"""Check configured image predictions and initialization pairs on a CUDA GPU."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# Support direct Play in PyCharm as well as python -m execution.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

import torch
import torchvision
from torchvision.models import ResNet50_Weights

from src.model_wrappers import TorchModelWrapper

from experiments.initialization_study.configuration.loader import STUDY_DIR, get_source_label, read_config
from experiments.initialization_study.inputs.images import image_path, read_images
from experiments.initialization_study.integration.model import load_target_model, require_cuda
from experiments.initialization_study.inputs.preprocessing import load_image_tensor
from experiments.initialization_study.configuration.schemas import SelectedGroup, StudyConfig
from experiments.initialization_study.integration.schemas import ImagePrediction, PairCheck, SourceCheck, SourceLabelStatus


################## INPUT VALIDATION ##################

"""
    Validates every group before loading model weights or creating output.
    @type config: StudyConfig
    @param config: The validated study configuration.
    @type selected: list[SelectedGroup]
    @param selected: The (family, group) pairs to check.
    @rtype: dict[str, tuple[list[Path], list[Path]]]
    @returns: Source and initialization paths for each group.
"""
def collect_groups(
    config: StudyConfig, selected: list[SelectedGroup]
) -> dict[str, tuple[list[Path], list[Path]]]:
    groups: dict[str, tuple[list[Path], list[Path]]] = {}

    for family, group in selected:
        settings = config["groups"][group]
        sources = read_images(settings["source"], STUDY_DIR, group + ".source")
        initializations = read_images(settings["initializations"], STUDY_DIR, group + ".initializations")
        groups[group] = (sources, initializations)

    return groups


################## PREDICTIONS AND PAIR CHECKS ##################

"""
    Predicts once per unique resolved image path, including shared group inputs.
    @type model: TorchModelWrapper
    @param model: The loaded GPU-only target model.
    @type groups: dict[str, tuple[list[Path], list[Path]]]
    @param groups: Validated source and initialization paths.
    @rtype: dict[Path, ImagePrediction]
    @returns: Cached predicted class IDs and names, keyed by absolute image path.
"""
def predict_images(
    model: TorchModelWrapper, groups: dict[str, tuple[list[Path], list[Path]]]
) -> dict[Path, ImagePrediction]:
    predictions: dict[Path, ImagePrediction] = {}
    device = next(model.parameters()).device
    categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    for sources, initializations in groups.values():
        for path in sources + initializations:
            if path in predictions:
                continue

            image = load_image_tensor(path).to(device)
            class_id = int(model.predict_label(image).item())
            prediction: ImagePrediction = {
                "path": image_path(path, STUDY_DIR),
                "class_id": class_id,
                "class_name": categories[class_id],
            }
            predictions[path] = prediction
            print("{}: {} ({})".format(prediction["path"], prediction["class_name"], class_id))

    return predictions


"""
    Compares every source prediction with its ground-truth label inferred from its class folder.
    Uses the numeric prefix of the source class folder without additional model queries.
    @type selected: list[SelectedGroup]
    @param selected: The family and group pairs to check.
    @type groups: dict[str, tuple[list[Path], list[Path]]]
    @param groups: Validated source and initialization paths.
    @type predictions: dict[Path, ImagePrediction]
    @param predictions: Cached predictions from the selected target model.
    @rtype: list[SourceCheck]
    @returns: One correctness record per source in each group.
"""
def check_sources(
    selected: list[SelectedGroup],
    groups: dict[str, tuple[list[Path], list[Path]]], predictions: dict[Path, ImagePrediction],
) -> list[SourceCheck]:
    checks: list[SourceCheck] = []

    for family, group in selected:
        sources, _ = groups[group]

        for source in sources:
            true_label = get_source_label(source)
            predicted_label = predictions[source]["class_id"]
            status: SourceLabelStatus = "correct" if predicted_label == true_label else "incorrect"
            checks.append({
                "family": family,
                "group": group,
                "source": predictions[source]["path"],
                "true_class_id": true_label,
                "predicted_class_id": predicted_label,
                "source_label_status": status,
            })
            print("{}/{}: {} | ground truth: {} | predicted: {} | {}".format(
                family, group, source.name, true_label, predicted_label, status,
            ))

    return checks


"""
    Checks whether each initialization changes the source's predicted class.
    Eligibility requires a correct source and an initialization changing its class.
    @type selected: list[SelectedGroup]
    @param selected: The family and group names for the report.
    @type groups: dict[str, tuple[list[Path], list[Path]]]
    @param groups: The validated input paths for each group.
    @type predictions: dict[Path, ImagePrediction]
    @param predictions: Cached model predictions for all inputs.
    @rtype: list[PairCheck]
    @returns: One classification-change check per source/initialization pair.
"""
def check_pairs(
    selected: list[SelectedGroup], groups: dict[str, tuple[list[Path], list[Path]]],
    predictions: dict[Path, ImagePrediction]
) -> list[PairCheck]:
    checks: list[PairCheck] = []

    for family, group in selected:
        sources, initializations = groups[group]

        for source in sources:
            for initialization in initializations:
                source_prediction = predictions[source]
                initialization_prediction = predictions[initialization]
                true_label = get_source_label(source)
                source_correct = source_prediction["class_id"] == true_label
                changes_class = source_prediction["class_id"] != initialization_prediction["class_id"]
                checks.append({
                    "family": family,
                    "group": group,
                    "source": source_prediction["path"],
                    "initialization": initialization_prediction["path"],
                    "source_class_id": source_prediction["class_id"],
                    "initialization_class_id": initialization_prediction["class_id"],
                    "changes_source_prediction": changes_class,
                    "source_true_class_id": true_label,
                    "source_label_status": "correct" if source_correct else "incorrect",
                    "eligible": source_correct and changes_class,
                })
                print("{}/{}: {} -> {} | changes class: {} | eligible: {}".format(
                    family, group, source.name, initialization.name, changes_class, source_correct and changes_class
                ))

    return checks


################## COMMAND LINE ##################

"""
    Performs GPU-only preparation checks and saves a separate JSON report.
    Checks source predictions against class-folder labels before pair eligibility.
    @type argv: list[str] or None
    @param argv: Optional arguments; None uses the command-line arguments.
"""
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Path relative to the study directory or absolute.")
    parser.add_argument("--device", default="cuda", help="CUDA GPU, for example cuda:0. CPU is disabled.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = STUDY_DIR / config_path

    try:
        # Check GPU availability before any model creation or weight download.
        device = require_cuda(args.device)
        config, selected = read_config(config_path)
        groups = collect_groups(config, selected)
        model = load_target_model(str(device))

        predictions = predict_images(model, groups)
        source_checks = check_sources(selected, groups, predictions)
        checks = check_pairs(selected, groups, predictions)
        report = {
            "config": config,
            "config_path": str(config_path.resolve()),
            "model": "resnet50",
            "weights": "IMAGENET1K_V1",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "preprocessing": ["RGB", "Resize(256)", "CenterCrop(224)", "ToTensor", "ImageNet normalization"],
            "preparation_queries": model.num_queries,
            "attack_queries": 0,
            "source_label_status": "checked",
            "source_label_origin": "class_directory",
            "sources": source_checks,
            "predictions": list(predictions.values()),
            "pairs": checks,
        }

        stamp = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d_%H-%M-%S-%f")
        output_dir = STUDY_DIR / "preflight_reports" / ("preflight-" + stamp)
        output_dir.mkdir(parents=True, exist_ok=False)
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ZoneInfoNotFoundError) as error:
        parser.exit(1, "Preflight failed: " + str(error) + "\n")

    print("\nPreparation queries: {} | Attack queries: 0".format(model.num_queries))
    print("Correct sources: {} / {}".format(
        sum(source["source_label_status"] == "correct" for source in source_checks), len(source_checks),
    ))
    print("Report: " + str(report_path))


if __name__ == "__main__":
    main()
