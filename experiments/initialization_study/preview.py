"""Save original/preprocessed image comparisons without loading a model."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image, ImageDraw
from torchvision import transforms

from .configuration import STUDY_DIR, read_config
from .images import read_images
from .validation import PlanError


################## INPUTS ##################

"""
    Validates all configured images before creating any preview output.
    @type config: dict
    @param config: The validated study configuration.
    @type selected: list
    @param selected: The (family, group) pairs to preview.
    @rtype: list
    @returns: Tuples of (family, group, role, image path), in configuration order.
"""
def collect_images(config, selected):
    entries = []

    for family, group in selected:
        settings = config["groups"][group]

        for role in ("source", "initializations"):
            paths = read_images(settings[role], STUDY_DIR, group + "." + role)

            for path in paths:
                entries.append((family, group, role, path))

    return entries


################## IMAGE COMPARISON ##################

"""
    Places the full original beside the resized and cropped image.
    The original is reduced for display only; the processed image stays 224x224.
    @type original: PIL.Image.Image
    @param original: The full RGB image before model preprocessing.
    @type processed: PIL.Image.Image
    @param processed: The RGB image after Resize(256) and CenterCrop(224).
    @rtype: PIL.Image.Image
    @returns: A labeled comparison image with the original on the left.
"""
def create_comparison(original, processed):
    original_display = original.copy()
    original_display.thumbnail((512, 512))

    # Keep enough room for labels even when the original is very narrow.
    margin = 20
    header_height = 60
    left_width = max(original_display.width, 224)
    right_width = processed.width
    image_height = max(original_display.height, processed.height)
    canvas_width = left_width + right_width + 3 * margin
    canvas_height = header_height + image_height + margin
    comparison = Image.new("RGB", (canvas_width, canvas_height), "white")

    left_position = (margin + (left_width - original_display.width) // 2, header_height)
    right_position = (left_width + 2 * margin, header_height)
    comparison.paste(original_display, left_position)
    comparison.paste(processed, right_position)

    labels = ImageDraw.Draw(comparison)
    labels.text((margin, 12), "Original (full image)", fill="black")
    labels.text((margin, 30), "{} x {}".format(*original.size), fill="black")
    labels.text((right_position[0], 12), "Resize(256) + CenterCrop(224)", fill="black")
    labels.text((right_position[0], 30), "224 x 224", fill="black")

    return comparison


"""
    Creates and saves one comparison for each configured image entry.
    Uses the same torchvision spatial transforms as src/dataset.py.
    @type entries: list
    @param entries: Validated (family, group, role, image path) tuples.
    @type output_dir: Path
    @param output_dir: The new preview directory to create.
    @returns: None. Prints the input filename and comparison path for each entry.
"""
def save_previews(entries, output_dir):
    spatial_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    # A new directory for each invocation keeps previous previews intact.
    output_dir.mkdir(parents=True, exist_ok=False)

    for family, group, role, path in entries:
        with Image.open(path) as image:
            original = image.convert("RGB")

        # Preview only the spatial processing; tensors and normalization are not display images.
        processed = spatial_transform(original)
        comparison = create_comparison(original, processed)

        destination = output_dir / family / group / role / (path.stem + "-preview.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(destination)

        print(path.name + " -> " + str(destination))


################## COMMAND LINE ##################

"""
    Reads the configuration and saves image comparisons when explicitly invoked.
    No models, model weights, predictions, or attacks are used.
    @type argv: list or None
    @param argv: Optional arguments; None uses the command-line arguments.
"""
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json",
                        help="Absolute path or path relative to the study directory.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = STUDY_DIR / config_path

    try:
        config, selected = read_config(config_path)
        entries = collect_images(config, selected)

        stamp = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d_%H-%M-%S-%f")
        output_dir = STUDY_DIR / "previews" / ("preview-" + stamp)
        save_previews(entries, output_dir)
    except (PlanError, OSError, json.JSONDecodeError, ZoneInfoNotFoundError) as error:
        parser.exit(1, "Preview failed: " + str(error) + "\n")

    print("\nSaved {} comparisons in: {}".format(len(entries), output_dir))


if __name__ == "__main__":
    main()
