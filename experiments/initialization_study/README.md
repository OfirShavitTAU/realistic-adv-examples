# Initialization study

This directory contains our experiment layer. The original `src` and `main.py` are unchanged.

## Current stage: plan only

Run from the repository root, using the repository's virtual environment:

```powershell
.\venv\Scripts\python.exe -m experiments.initialization_study.run
```

This validates the JSON settings, decodes the selected images, and lists every planned
source/initialization pair plus one random baseline per source, norm, and seed.
It does **not** import PyTorch, download weights, run attacks, or create results.
Pillow is used to verify images and is already a dependency of the existing environment.

Edit `config.json`: `experiment_groups` mirrors the folder hierarchy. Each family
contains group names with `true` (selected) or `false` (disabled). Unlisted groups
are not selected. For example, add `"group_2": true` inside `"dogs"` to select it.
The default budgets are small pilot settings, not the paper's full evaluation budgets.

Each selected group must contain image files directly in both directories:

```text
experiment_groups/<family>/<group>/source/
experiment_groups/<family>/<group>/initializations/
```

Every source is paired with every initialization in its own group. Groups never
cross-pair. Supported extensions: JPG, JPEG, JFIF, PNG, BMP, WEBP (case insensitive).
Duplicate image stems in a directory are rejected to prevent output collisions.
The stem `random` is reserved for the random baseline. Empty selected groups are errors.

Alternative configuration:

```powershell
.\venv\Scripts\python.exe -m experiments.initialization_study.run --config configs/pilot.json
```

Relative configuration paths are resolved against this study directory. Images and
results always use this study directory, regardless of the config file location.
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
