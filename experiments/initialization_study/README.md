# Initialization study

This directory contains our experiment layer. The original `src` and `main.py` are unchanged.

## Code structure

```text
initialization_study/
    config.json
    run/                         # Tool implementations; press Play on the chosen file
        run_preview.py           # Play to create image comparisons
        run_preflight.py         # Play for GPU-only model preparation checks
        run_experiment.py        # Plan by default; --execute runs the GPU pipeline
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
        hsja.py                  # Call original HSJA with an image or random initialization
        pipeline.py              # Load once, check inputs, seed and execute tasks
        schemas.py               # Prediction and preparation report types
    images/                      # Existing input images
    previews/                    # Generated comparisons (ignored by Git)
    preflight_reports/           # Generated preparation reports (ignored by Git)
```

Each code subdirectory has an `__init__.py` package marker. The three runnable
tools contain their own implementations under `run/`; there are no forwarding
scripts or separate `pre_run` directory. Relative configuration and image paths
still resolve against `initialization_study`, not against the `run` directory.
`integration/hsja.py` now delegates single-task calls to the original HSJA.
The experiment runner plans by default and calls the GPU pipeline only with `--execute`.

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
The study requirements include Foolbox for the original attack and torch_dct because
`src.attacks.__init__` imports GeoDA even when selecting HSJA. The full root requirements
are still needed for unrelated research entry points and models.

## Planning (default mode)

Run from the repository root, using the repository's virtual environment:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_experiment
```

This validates the JSON settings, decodes the selected images, and lists every planned
source/initialization pair plus one random baseline per source, norm, and seed.
It imports the pipeline dependencies but does **not** load a model, download weights,
run attacks, or create results unless execution is explicitly requested.
Pillow is used to verify images and is included in the study requirements.

Edit `config.json`: `groups` defines the groups directly, with a `family`, a
nonempty `source` list and a nonempty `initializations` list of image paths:

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

## Automatic source labels from class folders

The configuration contains only image paths, not numeric source labels. The code
reads each source's zero-based ImageNet class ID from the numeric prefix of its
immediate parent directory, for example:

```text
images/dogs/207_golden_retriever/ILSVRC2012_val_00034584.jfif
            ^ label 207
```

Use `<ImageNet ID>_<class name>` folders with IDs from 0 to 999. Missing prefixes,
invalid IDs, and missing class names are rejected before model loading. Sources
must be directly inside their class folders, not nested in unlabeled subdirectories.
Relative paths are resolved against the study directory; symlinks use the resolved
file's actual class folder. Original ImageNet IDs are preserved without renumbering.
No image moves or separate source mapping file are needed for the existing layout.

Both preflight and the pipeline compare each source's prediction with this folder
label. Incorrectly classified sources are skipped for all tasks, including random
baselines. Correct sources use the inferred true class as HSJA's original label.
Initialization images need a different prediction; they do not require true labels.

The directory assignment supplies the dataset ground truth. The code does not
independently verify the human annotation or the meaning of the text after the ID.
Preflight reports record `source_label_origin` as `class_directory`.

## Direct Play in PyCharm

Open one of these files under `experiments/initialization_study/run/`, select the
project's Python 3.10 virtual environment, and press Play:

| File | Action |
| --- | --- |
| `run_preview.py` | Save original/cropped image comparisons; no GPU needed. |
| `run_preflight.py` | Check predictions and initialization pairs; CUDA GPU required. |
| `run_experiment.py` | Display the plan by default; add `--execute` in run parameters to execute on GPU. |

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

The adapter now calls the original `HSJA.hsja` with the configured initialization
through `target_image`, or None for the original random baseline, keeping
`target_label=None` for untargeted execution. Gradient estimation, searches,
updates, query counting, and stopping remain in the original attack code.

## Original HSJA adapter

`integration/hsja.py` exposes `create_hsja(settings, norm)` and
`run_hsja(model, source, source_label, settings, norm, initialization=None)`.
A fresh original attack is created per call. Both image inputs use shape
`(1, 3, 224, 224)`, float32 raw pixels in [0, 1], on the model's CUDA GPU; the adapter
removes only the initialization batch dimension to match the internal HSJA interface.
The label is a `torch.long` tensor of shape `(1,)` on the same GPU.

The adapter maps configured budgets and iterations to the original constructor.
Other settings use the original defaults: continuous pixels, [0, 1] bounds,
binary boundary search, HSJA gradient estimation, and geometric step-size search.
It forwards those settings explicitly to the internal method, matching the original
`__call__` path. It returns the original five-element tuple unchanged.

Automatic class-folder labels and source-correctness validation are now
enforced by the pipeline. The low-level adapter still expects its caller to check
source and initialization predictions; shape/device checks do not replace eligibility.
Seed setup and orchestration are connected in `integration/pipeline.py`. Independent
result verification, effective-parameter recording, and result persistence remain pending.
Original budgets can be exceeded because HSJA checks them at iteration boundaries;
the returned success flag is not independent verification or a distance threshold.

`run/run_experiment.py` prints the plan by default; `--execute` invokes the adapter
through the pipeline. Importing the adapter does not load a model or run an attack. Isolated delegation checks use mocked tensors and
replace the original optimization method; they require no GPU and execute no model:

```bash
.venv/bin/python -m unittest experiments.initialization_study.tests.test_hsja_adapter
```

## GPU execution pipeline (stage 3)

Execution and automatic class-folder label checks are connected. Independent final-result
verification and persistence (step 4) remain pending. The default remains planning only.
To explicitly execute on the GPU VM, use:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_experiment --execute --device cuda:0
```

In PyCharm, the same options can be entered in the script's run parameters. No
execution is triggered merely by importing a module or running without `--execute`.

The pipeline performs these steps:

1. Require the selected CUDA GPU, then validate group image files.
2. Load the original model wrapper once for the whole experiment.
3. Reuse the standalone preflight functions to predict each unique image once and
   compare each source prediction with its class-folder ground truth, and check pair
   eligibility. Count preparation queries separately.
4. Iterate over the already planned tasks, preserving group, norm, and seed choices.
5. Skip every task of a source predicted incorrectly, including all its random
   baselines, norms, and seeds. For correct sources, skip only image initializations
   predicted as the source class; random baselines remain eligible.
6. For each eligible task, load raw tensors onto the GPU, reset actual model-query
   counting, and seed Python, NumPy, PyTorch, and CUDA with the task's seed.
7. Call the adapter, which constructs a fresh original HSJA instance. Image and
   random initialization paths stay untargeted.
8. Keep the exact returned output pixels on CPU to avoid accumulating image tensors
   on the GPU between tasks, while retaining original counters and the success flag.
9. Print per-task status, actual model queries, elapsed task time, and a final summary.

Task time includes tensor preparation and device synchronization, not just the HSJA
call. Seeds are repeatable on a fixed setup, not a guarantee across different GPUs
or software versions. Expected task failures are recorded with their consumed queries
and the pipeline proceeds to later tasks; KeyboardInterrupt is not swallowed.
Setup failures stop the pipeline. The command exits with an error status if any task
failed, but skipped tasks are not errors.

**Current limitations:** a `completed` task means HSJA returned, not independently
verified attack success. Attack reference labels are the class-folder true labels, after
checking that cached source predictions match them. The original counter and actual model-query count are both
available, and may differ; final accounting checks are pending. No result files,
tensors, images, or summary CSV are saved yet. CLI outcomes are lost on process exit.
Programmatic callers can retain the returned `PipelineResult` from `run_pipeline`.
Full experimental runs should wait for the persistence stage.

The isolated adapter and pipeline checks execute mocked calls only:

```bash
.venv/bin/python -m unittest experiments.initialization_study.tests.test_hsja_adapter experiments.initialization_study.tests.test_pipeline experiments.initialization_study.tests.test_source_labels
```

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

Source correctness is checked against the IDs inferred from source class folders. Each source
is marked `correct` or `incorrect` and records both true and predicted ImageNet IDs.
Each pair records `eligible`, which requires both a correct source and an
initialization predicted as another class. Preflight reports mismatches; the pipeline
skips all tasks associated with incorrectly predicted sources. These comparisons
reuse cached predictions and add no model queries.

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
the image tree. Attack execution is connected with `--execute`; this result storage
and independent final-result verification are not implemented yet.
