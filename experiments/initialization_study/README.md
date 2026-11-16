# Initialization study

This directory contains our experiment layer. The original `src` and `main.py` are unchanged.

## Code structure

- `run.py`: command-line entry point that connects the planning steps.
- `configuration.py`: study directory and JSON configuration validation.
- `validation.py`: shared validation helpers and `PlanError`.
- `images.py`: image validation and path formatting.
- `planner.py`: source/initialization pairing and planned output paths.
- `display.py`: terminal output for the plan.

## Current stage: plan only

Run from the repository root, using the repository's virtual environment:

```bash
.venv/bin/python -m experiments.initialization_study.run
```

This validates the JSON settings, decodes the selected images, and lists every planned
source/initialization pair plus one random baseline per source, norm, and seed.
It does **not** import PyTorch, download weights, run attacks, or create results.
Pillow is used to verify images and is already a dependency of the existing environment.

Edit `config.json`: `groups` defines the groups directly, with a `family`, a
nonempty `source` list, and a nonempty `initializations` list of image paths:

```json
{
  "groups": {
    "group_1": {
      "family": "dogs",
      "source": ["images/dogs/207_golden_retriever/ILSVRC2012_val_00034584.jfif"],
      "initializations": ["images/dogs/208_labrador_retriever/ILSVRC2012_val_00019169.jfif"]
    }
  }
}
```

This snippet shows only the group definition; keep `random_baseline` and `attack`
as in the complete `config.json`. The checked-in configuration contains example
image selections, which you can replace with your intended study inputs.
All listed groups are included. Add another entry such as `group_2` to define another
group; remove an entry to exclude it. No `experiment_groups` directories or image
copies are needed. The same image can be referenced in multiple groups.
Relative image paths are resolved against this study directory, even when using
an alternative config. Absolute image paths are also supported.

Every source is paired with every initialization in its own group. Groups never
cross-pair. Supported extensions: JPG, JPEG, JFIF, PNG, BMP, WEBP (case insensitive).
Duplicate image stems within either list of a group are rejected to prevent output
collisions, even if the files are in different directories. The stem `random` is
reserved for the random baseline. Missing or corrupt images are rejected.
The default budgets are small pilot settings, not the paper's full evaluation budgets.

Alternative configuration:

```bash
.venv/bin/python -m experiments.initialization_study.run --config configs/pilot.json
```

Relative configuration paths are resolved against this study directory. Relative image paths and
results use this study directory, regardless of the config file location.
For module imports, run the command from the repository root.

## Output contract for the execution stage

Actual runs will create `results/experiment-YYYY-MM-DD_HH-MM-SS-microseconds/` using
`Asia/Jerusalem` time. The planner displays a preview name only. The execution stage
will generate the timestamp when execution starts. Windows can sort these names
chronologically. On Windows, timezone support may require the `tzdata` package.

```text
results/experiment-<timestamp>/
    config.json
    run_info.json
    summary.csv
    data/...
    images/<family>/<group>/
        source/<source files>
        initializations/<initialization files>
        results/<source stem>/<norm>_seed_<seed>/
            <initialization stem>-atttack.png
            random-atttack.png
```

The spelling `-atttack` is intentional. Norm/seed subdirectories prevent overwrites.
Source labels and actual model predictions are separate from folder names. Execution
must validate source labels and ensure initializations change the original class.
Per-attack metrics, precise tensors, and query traces will be kept separately from
the image tree. None of those execution features are implemented at this stage.
