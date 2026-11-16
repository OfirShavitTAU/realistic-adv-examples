"""Validate image files and format their paths."""

from pathlib import Path

from PIL import Image

from .validation import PlanError, folder_name, require


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}


################## IMAGES AND PATHS ##################

"""
    Resolves image paths and checks files, names, and image contents.
    @type paths: list
    @param paths: The configured image paths, in their original order.
    @type study_dir: Path
    @param study_dir: The base directory for relative image paths.
    @type context: str
    @param context: The group and role used in validation error messages.
    @rtype: list
    @returns: The absolute paths of the validated images.
    @raises PlanError: If images are missing, corrupt, or have conflicting stems.
"""
def read_images(paths: list[str], study_dir: Path, context: str) -> list[Path]:
    images: list[Path] = []

    for value in paths:
        path = Path(value)

        if not path.is_absolute():
            path = study_dir / path

        path = path.resolve()
        require(path.is_file(), "Missing image in " + context + ": " + str(path))
        require(path.suffix.lower() in IMAGE_EXTENSIONS, "Unsupported image extension: " + str(path))
        images.append(path)

    # Results use image stems as directory/file names, so extensions cannot distinguish them.
    stems = [path.stem.casefold() for path in images]
    require(len(set(stems)) == len(stems), "Images with duplicate stems would overwrite results: " + context)

    for path in images:
        folder_name(path.stem)
        require(path.stem.casefold() != "random", "The image stem 'random' is reserved for the baseline.")

        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as exc:
            raise PlanError("Cannot decode image: " + str(path)) from exc

    return images


"""
    Formats an image path for the plan.
    @type path: Path
    @param path: The resolved image path.
    @type study_dir: Path
    @param study_dir: The study directory used for relative paths.
    @rtype: str
    @returns: A relative path inside the study, otherwise an absolute path.
"""
def image_path(path: Path, study_dir: Path) -> str:
    try:
        return path.relative_to(study_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
