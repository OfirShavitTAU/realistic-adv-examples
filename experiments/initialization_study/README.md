# Initialization study

This directory contains our experiment layer. The original `src` and `main.py` are unchanged.

## Code structure

```text
initialization_study/
    config.json
    run/                         # Tool implementations; press Play on the chosen file
        run_preview.py           # Play to create image comparisons
        run_preflight.py         # Model preparation checks; --cpu explicitly selects CPU
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
        model.py                 # Load the original model wrapper; CUDA default, --cpu optional
        hsja.py                  # Call original HSJA with an image or random initialization
        verification.py          # Independently classify exact final output pixels
        pipeline.py              # Load once, check inputs, seed and execute tasks
        schemas.py               # Prediction and preparation report types
    outputs/
        writer.py                # Run snapshots, artifacts, and incremental summary
        serialization.py         # Atomic JSON, CSV, and exact tensor writes
    images/                      # Existing input images
    previews/                    # Generated comparisons (ignored by Git)
    preflight_reports/           # Generated preparation reports (ignored by Git)
    results/                     # Generated experiment runs (ignored by Git)
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
The study requirements cover planning, previews, and preflight predictions:
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
| `run_preflight.py` | Check predictions and initialization pairs; CUDA default, `--cpu` for CPU. |
| `run_experiment.py` | Display the plan by default; add `--execute` in run parameters to execute on GPU. |

There is no menu or code setting to change. Each file contains its tool's own
implementation and uses this study's `config.json` by default. The tools support
direct file execution even with a different working directory and accept their
underlying tool's command-line arguments. Model inference defaults to CUDA; CPU requires `--cpu`.

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
`(1, 3, 224, 224)`, float32 raw pixels in [0, 1], on the model's selected device; the adapter
removes only the initialization batch dimension to match the internal HSJA interface.
The label is a `torch.long` tensor of shape `(1,)` on the same device.

The adapter maps configured budgets and iterations to the original constructor.
Other settings use the original defaults: continuous pixels, [0, 1] bounds,
binary boundary search, HSJA gradient estimation, and geometric step-size search.
It forwards those settings explicitly to the internal method, matching the original
`__call__` path. It returns the original five-element tuple unchanged.

Automatic class-folder labels and source-correctness validation are now
enforced by the pipeline. The low-level adapter still expects its caller to check
source and initialization predictions; shape/device checks do not replace eligibility.
Seed setup, orchestration, and incremental storage are connected in `integration/pipeline.py`.
Effective initial parameters are saved with each run. Final classification is verified
independently; distance verification and query-accounting validation remain pending.
Original budgets can be exceeded because HSJA checks them at iteration boundaries;
the returned success flag is not independent verification or a distance threshold.

`run/run_experiment.py` prints the plan by default; `--execute` invokes the adapter
through the pipeline. Importing the adapter does not load a model or run an attack. Isolated delegation checks use mocked tensors and
replace the original optimization method; they require no GPU and execute no model:

```bash
.venv/bin/python -m unittest experiments.initialization_study.tests.test_hsja_adapter
```

## Execution pipeline (stage 3)

Execution, automatic class-folder label checks, and incremental storage are connected.
Final classification is independently verified. Distance verification and query-accounting
validation remain pending. The default remains planning only.
To explicitly execute on the GPU VM, use:

```bash
.venv/bin/python -m experiments.initialization_study.run.run_experiment --execute --device cuda:0
```

In PyCharm, the same options can be entered in the script's run parameters. No
execution is triggered merely by importing a module or running without `--execute`.

The pipeline performs these steps:

1. Require the selected CUDA GPU unless `--cpu` is explicit, validate inputs, and create a new output
   directory with configuration, parameter/environment metadata, and pending summary rows.
2. Copy the original input files into the output tree, then load the original model
   wrapper once for the whole experiment.
3. Reuse the standalone preflight functions to predict each unique image once and
   compare each source prediction with its class-folder ground truth, and check pair
   eligibility. Save the preparation report and count its queries separately.
4. Iterate over the already planned tasks, preserving group, norm, and seed choices.
5. Skip every task of a source predicted incorrectly, including all its random
   baselines, norms, and seeds. For correct sources, skip only image initializations
   predicted as the source class; random baselines remain eligible.
6. For each eligible task, load raw tensors onto the selected device, reset actual model-query
   counting, and seed Python, NumPy, and PyTorch with the task's seed (also CUDA in GPU mode).
7. Call the adapter, which constructs a fresh original HSJA instance. Image and
   random initialization paths stay untargeted.
8. Keep the exact returned output pixels on CPU while retaining original counters
   and the success flag. Reclassify the exact output pixels and record verified success
   or failure, counting the extra query separately.
   Save exact source/initialization tensors, output tensors, display PNGs, and query history.
9. Update the summary CSV and run state after every task, including skipped and failed
   tasks, then print a final summary and the output directory.

Task time includes tensor preparation and, in GPU mode, device synchronization, not just the HSJA
call. Seeds are repeatable on a fixed setup, not a guarantee across different GPUs
or software versions. Expected task failures are recorded with their consumed queries
and the pipeline proceeds to later tasks; KeyboardInterrupt is not swallowed.
Setup failures stop the pipeline. The command exits with an error status if any task
failed, but skipped tasks are not errors.

**Current limitations:** execution status `completed` means the algorithm returned and
final classification was checked; consult `attack_outcome` for success or failure.
Success currently means a final class different from the verified source class, with
no distance threshold. The original counter and actual attack-query count are both
available and may differ; final accounting checks are pending. Independent L2/Linf
fields remain blank with `distance_validation_status=not_checked`. Programmatic callers also receive a `PipelineResult` containing outcomes
and the saved output directory. GPU execution still needs an actual pilot run.

The isolated adapter and pipeline checks execute mocked calls only (no real model or attack):

```bash
.venv/bin/python -m unittest experiments.initialization_study.tests.test_hsja_adapter experiments.initialization_study.tests.test_pipeline experiments.initialization_study.tests.test_source_labels
```

## Model preflight

Model inference defaults to CUDA, with no automatic CPU fallback. Without `--cpu`,
unavailable CUDA or an invalid GPU index stops execution before model construction,
weight downloads, or creation of an experiment output directory. This checks whether
PyTorch can use the GPU, not merely whether hardware is physically installed.

For the GPU VM, use CUDA-enabled torch/torchvision matching the version constraints
and a compatible NVIDIA driver. Select a GPU with `--device cuda:0`. CPU mode is
explicitly supported for small end-to-end checks on the development VM:

```bash
# Preparation predictions only, explicitly on CPU:
.venv/bin/python -m experiments.initialization_study.run.run_preflight --cpu

# Actual attack pipeline with per-task storage, explicitly on CPU:
.venv/bin/python -m experiments.initialization_study.run.run_experiment --execute --cpu
```

In PyCharm, put `--cpu` (and `--execute` for the experiment) in the Run configuration's
Parameters field. Without those options, pressing Play on the experiment still plans
only. `--cpu` and `--device` are mutually exclusive. Preview needs neither option.
The first model load on either device may download V1 weights if not cached.
No CPU fallback occurs when a GPU run fails.

The original model wrapper's constructor hardcodes `.cuda()` for supplied normalization
statistics. Our loader constructs that same wrapper without statistics, then assigns
the original mean/std on the selected device. Its preprocessing, prediction and query
counting methods are reused unchanged; no researchers' source files are edited.
CPU runs use CPU RNG seeding and skip CUDA synchronization and CUDA-specific settings.
Run metadata records the actual selected device, and CPU reports have no GPU name.

The target is torchvision ResNet-50 with `ResNet50_Weights.IMAGENET1K_V1` in eval
mode, without gradient tracking or DataParallel. Images use RGB conversion,
`Resize(256)`, `CenterCrop(224)`, and `ToTensor`. Raw tensors remain in [0, 1];
The original `TorchModelWrapper` handles ImageNet normalization, label prediction,
and query counting. Those operations are not reimplemented in our code. File decoding and
image transforms use the host CPU; all network inference runs on the explicitly selected device.

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

## Saved experiment output

Executing with `--execute` creates a new directory using Israel time. Plan-only mode
creates no output. An existing run directory is never reused or overwritten.

```text
results/
    experiment-YYYY-MM-DD_HH-MM-SS-microseconds/
        config.json
        run_info.json
        summary.csv
        data/
            plan.json
            input_manifest.json
            preflight.json
            task-000001/
                task.json
                source.pt
                initialization.pt
                output.pt
                queries.json
            task-000002/...
        images/<family>/<group>/
            source/<original files>
            initializations/<original files>
            results/<source stem>/<norm>_seed_<seed>/
                <initialization stem>-atttack.png
                random-atttack.png
```

`config.json` snapshots the parsed configuration. `run_info.json` records the selected
model/weights, package versions, Git commit and dirty state, device information,
effective initial HSJA parameters, timestamps, preparation cost, and execution state.
`input_manifest.json` records original paths, copied paths, and SHA-256 hashes. Input
copies preserve the original bytes. `preflight.json` stores cached predictions,
source correctness, pair eligibility, and preparation queries.

Stable task IDs match plan order. `task.json` records the planned task, status,
reason/error, original extra results, artifact links, and validation status. Exact
raw tensors are saved with `torch.save` on CPU; they are not ImageNet-normalized.
Source and initialization tensors contain the pixels supplied to the original attack.
`output.pt` preserves the returned pixels exactly; display PNGs alone are clipped and
rounded to 8-bit RGB. The intentional `-atttack` suffix and norm/seed folders are kept.
Random tasks have no external `initialization.pt`; the original algorithm does not
return its initial random noise. Skipped tasks have metadata without fake outputs.
Errors retain any prepared input tensors and actual queries consumed. Original
query history is marked unavailable if an exception prevents HSJA returning it.

`queries.json` preserves original phase counts and distance history, including safe
flags, best distances, and equivalent simulated-query counts. Nonfinite JSON numbers
are represented by null. `summary.csv` uses UTF-8 with a BOM and stores input IDs,
source and initialization predictions, norm/seed, status/reason, task time, actual
model queries, reported total/unsafe/safe queries, reported distance/success, and
artifact paths. Safe/unsafe names retain the researchers' semantics. Preparation
cost is stored once per run, not duplicated into every task's attack cost.

Each returned result is independently classified from the exact tensor before PNG
rounding. `final_predicted_class_id` and `final_predicted_class_name` record the final
ImageNet class. `attack_success` is true when it differs from the source's verified
folder label. A checked same-class result is a verified failure, even if HSJA returned
its own success flag as true; that flag remains separately named `attack_reported_success`.
This checks classification only; independent distance computation and thresholds remain
pending. The distance fields stay blank with `distance_validation_status=not_checked`.

`attack_outcome` identifies each task as `success`, `failure`, `skipped`, `error`, or
`not_checked`; pending/running states appear while a run is in progress. A failure
means a completed, verified result retaining the source class. Skips and errors are
not counted as verified failures. If final inference errors, its result is unknown,
validation is marked `error`, and the original attack artifacts are still saved.
Task metadata and the CSV retain validation errors and the separate validation cost.

`run_info.json` stores outcome counts, task IDs grouped by outcome, verified-task
count, and `verified_success_rate = successes / (successes + failures)`. Skips,
errors, and unverified results do not enter this denominator; the rate is null if
there are no verified results. These outcomes and the final class are also printed
to the console. A run can complete normally while containing verified failures;
CLI errors indicate execution/validation errors, not simply a failed attack.

Each normal final check adds one model query. `actual_model_queries` remains the
attack-only cost, `validation_queries` is separate, and `total_task_model_queries`
includes both. Run-level `total_model_queries` also includes preparation queries.
Original HSJA phase histories and counters are not modified by final validation.
Older result directories are not retroactively classified or rewritten.

Each task is marked `running` before it starts. Artifacts are saved before publishing
its terminal summary row. JSON, tensor, and CSV replacements are atomic per file;
there is no transaction spanning all files. A graceful interruption marks the run
`interrupted`; setup/storage failures mark it `failed`. Unexpected termination may
leave the active task `running` with later tasks `pending`, while earlier completed
artifacts remain available. A task interrupted before returning cannot provide its
final tensor or full query history. Storage failures stop further execution rather
than silently discarding results. Automatic resume is not implemented.

Storage checks use synthetic CPU tensors, mocked model/attack calls, and temporary
directories only; they do not execute a real model or HSJA:

```bash
.venv/bin/python -m unittest discover -s experiments/initialization_study/tests
```


## Opt-in real CPU E2E

`tests/e2e/e2e.json` defines one golden-retriever source, one Labrador initialization,
L2, seed 0, one HSJA iteration, and query budgets of 100. Random baseline is disabled.
The researchers' budgets are checked at iteration boundaries, so these are not hard
query limits and do not guarantee a short runtime.

To run the real test, execute `tests/e2e/test_e2e.py` as a Python script in PyCharm
or run:

```bash
.venv/bin/python experiments/initialization_study/tests/e2e/test_e2e.py
```

The test starts the public experiment command in a separate Python process with
`--config <the E2E fixture> --execute --cpu`. It may download the V1 weights on first
use. It has a 600-second process timeout and retains normal experiment artifacts in
`results/experiment-<timestamp>` for inspection, including when assertions fail.
A process timeout may leave partial output.

It requires one completed task, CPU metadata, a summary row, a display PNG, exact
output/input tensors, and recorded query history. A skipped source or initialization
fails the test because it does not exercise the attack path. It also requires a checked final class, an explicit success/failure outcome, and
one separately counted validation query. It accepts either verified classification
outcome; this is an integration test, not a guarantee that one iteration succeeds.
Independent distance verification remains pending.

The `e2e` directory deliberately has no package marker so ordinary unittest discovery
does not recurse into it. Import-based test collection also skips the test unless
`RUN_INITIALIZATION_E2E=1` is explicitly set. If PyCharm selects a unittest run
configuration for this file, add that environment variable to enable the real test.
For explicit unittest execution:

```bash
RUN_INITIALIZATION_E2E=1 .venv/bin/python -m unittest discover -s experiments/initialization_study/tests/e2e
```
