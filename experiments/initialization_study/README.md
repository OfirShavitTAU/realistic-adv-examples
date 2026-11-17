# Initialization study

This directory contains our experiment layer. The original `src` and `main.py` are unchanged.

## Code structure

```text
initialization_study/
    config.json
    run/                         # Tool implementations; press Play on the chosen file
        run_preview.py           # Play to create image comparisons
        run_preflight.py         # Play for GPU-only model preparation checks
        run_experiment.py        # Play to display the experiment plan
    validation.py                # Shared checks and PlanError
    requirements.txt
    README.md
    configuration/
        loader.py                # Read and validate JSON; defines the study directory
        schemas.py               # Configuration and group types
    planning/
        planner.py               # Build source/initialization jobs
        schemas.py               # Planned-job types
        display.py               # Print the plan
    inputs/
        images.py                # Validate image files and format paths
        preprocessing.py         # Shared resize/crop and raw tensor preparation
        schemas.py               # Image preview entry types
    integration/
        model.py                 # Load the original TorchModelWrapper on a CUDA GPU
        schemas.py               # Prediction and preparation report types
    images/                      # Existing input images
    previews/                    # Generated comparisons (ignored by Git)
    preflight_reports/           # Generated preparation reports (ignored by Git)
```

Each code subdirectory has an `__init__.py` package marker. The three runnable
tools contain their own implementations under `run/`; there are no forwarding
scripts or separate `pre_run` directory. Relative configuration and image paths
still resolve against `initialization_study`, not against the `run` directory.
A future `integration/hsja.py` will call the original HSJA implementation; attack
execution is not connected yet.

## Python environment

Use Python 3.10, matching the original research environment. From the repository
root, create a virtual environment if needed and install the study dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r experiments/initialization_study/requirements.txt
```

If `.venv` already exists with Python 3.10, use only the installation command.
On Windows, use `py -3.10 -m venv .venv` and `.venv\Scripts\python.exe`.
The study requirements cover planning, previews, and GPU-only preflight predictions:
Pillow for images, torch/torchvision for transforms, NumPy compatible with the
original project, and tzdata for Israel timestamps on systems without timezone data.
The NumPy, torch, and torchvision constraints match the root requirements file.
These study requirements do not replace the full research requirements for future
attack execution; those dependencies will be added when the attack runner is implemented.

## Current stage: plan only

Run from the repository root, using the repository's virtual environment:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_experiment
```

This validates the JSON settings, decodes the selected images, and lists every planned
source/initialization pair plus one random baseline per source, norm, and seed.
It does **not** import PyTorch, download weights, run attacks, or create results.
Pillow is used to verify images and is included in the study requirements.

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
.venv/bin/python -m experiments.initialization_study.run.run_experiment --config configs/pilot.json
```

Relative configuration paths are resolved against this study directory. Relative image paths and
results use this study directory, regardless of the config file location.
For module imports, run the command from the repository root.

## Direct Play in PyCharm

Open one of these files under `experiments/initialization_study/run/`, select the
project's Python 3.10 virtual environment, and press Play:

| File | Action |
| --- | --- |
| `run_preview.py` | Save original/cropped image comparisons; no GPU needed. |
| `run_preflight.py` | Check predictions and initialization pairs; CUDA GPU required. |
| `run_experiment.py` | Validate inputs and display the experiment plan; no attacks yet. |

There is no menu or code setting to change. Each file contains its tool's own
implementation and uses this study's `config.json` by default. The tools support
direct file execution even with a different working directory and accept their
underlying tool's command-line arguments. Preflight stays GPU-only.

They can also be run as modules from the repository root:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_preview
.venv/bin/python -m experiments.initialization_study.run.run_preflight --device cuda:0
.venv/bin/python -m experiments.initialization_study.run.run_experiment
```

These replace the previous `pre_run` commands and the old standalone `run.py`.
Output locations are unchanged.

## Image preprocessing preview

To generate comparisons explicitly, run from the repository root:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_preview
```

Use `--config configs/pilot.json` for an alternative configuration, resolved against
this study directory just like the planner. The command validates all configured
source and initialization images, then saves one comparison per entry under:

```text
previews/preview-<Israel timestamp>/<family>/<group>/<source or initializations>/<image stem>-preview.png
```

The filename identifies the input image. Each comparison shows the full original
on the left (reduced to fit within 512x512 for display) and the actual 224x224 crop
on the right. The right side uses torchvision `Resize(256)` and `CenterCrop(224)`,
as in `src/dataset.py`, after RGB conversion as used by the default ImageFolder
PIL loader. No tensor conversion or normalization is needed for this visual preview.
The preview command imports torchvision/PyTorch for transforms but loads no model,
downloads no weights, computes no predictions, and runs no attacks. No GPU is needed.
Original files are never modified. Each invocation creates a separate directory;
if saving fails partway through, that directory may contain partial output.
The planner command is unchanged and does not generate previews.

## Reuse of the research code

The study layer defines image groups, prepares inputs, and collects reports.
`integration/model.py` contains `load_target_model`, which instantiates torchvision ResNet-50 with the same V1 weights as
`src/setup.py` and returns the researchers' original `TorchModelWrapper` directly.
It does not define a replacement model class. The original wrapper performs
normalization, `predict_label`, and query counting.

We do not call `setup_model_and_data` because it also builds an ImageFolder loader,
whereas this study reads explicit image paths from the configuration. The small
loader reproduces the ResNet branch's construction without its DataParallel layer.
No files in `src` or the original `main.py` are changed.

Attack execution is not connected yet. It will call the original `HSJA.hsja`
with the configured initialization through `target_image`, or None for the original
random baseline, keeping `target_label=None` for untargeted execution. Gradient
estimation, searches, updates, and stopping remain in the original attack code.

## GPU-only model preflight

Model inference must run on a CUDA GPU. CPU fallback is disabled. The development
VM without a GPU can still use the planner and image preview, but cannot run model
preflight. On the GPU VM, use a CUDA-enabled torch/torchvision installation matching
the version constraints and a compatible NVIDIA driver.

Run only when ready on the GPU VM, from the repository root:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_preflight --device cuda:0
```

An alternative configuration can be passed with `--config`. The command rejects a
CPU device or unavailable CUDA GPU before constructing the model or downloading
weights. The first successful GPU invocation may download the V1 weights into the
standard torchvision cache. Importing these modules does not load the model.

The target is torchvision ResNet-50 with `ResNet50_Weights.IMAGENET1K_V1` in eval
mode, without gradient tracking or DataParallel. Images use RGB conversion,
`Resize(256)`, `CenterCrop(224)`, and `ToTensor`. Raw tensors remain in [0, 1];
The original `TorchModelWrapper` handles ImageNet normalization, label prediction,
and query counting. Those operations are not reimplemented in our code. File decoding and
image transforms use the host CPU; all network inference runs on the selected GPU.

All configured image paths are validated first. Each unique resolved image path
is then predicted once, even if referenced by multiple groups. The report includes
original zero-based ImageNet class IDs and names and checks whether each configured
initialization changes its source's predicted class. Norms and seeds do not repeat
these deterministic preparation predictions. Random baselines are not sampled.

**Source correctness is not yet verified:** the configuration has no explicit
source ground-truth labels. Reports mark `source_label_status` as `not_checked`.
A pair changing predicted class alone does not establish eligibility for a study
requiring correctly classified sources. Folder names are not treated as verified
labels. Explicit ground-truth labels and their validation remain a separate step.

Reports are saved under `preflight_reports/preflight-<Israel timestamp>/report.json`
with the configuration, model weights, device, software versions, predictions,
pair checks, and preparation query count (one per evaluated image, not per batch).
No attacks are executed; attack queries are zero. These are separate preparation
reports, not experiment results. Model and attack code in `src` remains unchanged.

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
the image tree. Attack execution and its result storage are not implemented at this stage.
