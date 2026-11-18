# Adversarial Black Box Attack Initialization Study

This repository runs adversarial attack experiments (such as HopSkipJumpAttack) to evaluate how different initialization images affect attack performance across grouped ImageNet classes.

## Repository Structure

```text
.
├── experiments/
│   └── initialization_study/
│       ├── images/                  # Input images organized by family and ImageNet class
│       │   └── {family}/
│       │       └── <imageNet_label>_<imagenet_name>/
│       ├── inputs/
│       ├── integration/
│       ├── outputs/
│       ├── planning/
│       ├── results/
│       ├── run/
│       ├── tests/
│       ├── config.json              # Main experiment configuration
│       ├── progress.py
│       └── validation.py
├── logs/                            # SLURM output and error logs
├── scripts/
├── src/
├── tests/
├── environment.yml
├── main.py
├── plot_dist_vs_queries.py
├── pyproject.toml
├── requirements.txt
├── run_cpu.slurm
├── run_experiment.slurm             # SLURM script for full attack runs
└── run_preview.slurm                # SLURM script for image preprocessing preview
```

---

## 1. Where to Place Images

Place all input images inside `experiments/initialization_study/images/`, grouped by **family** and **ImageNet class directory** (`<imageNet_label>_<imagenet_name>`):

```text
experiments/initialization_study/images/{family}/<imageNet_label>_<imagenet_name>/<image_file>
```

### Example Directory Layout

```text
experiments/
└── initialization_study/
    └── images/
        ├── dogs/
        │   ├── 229_old_english_sheepdog/
        │   │   └── dogs1_init_far_old_english_sheepdog_ILSVRC2012_val_00030240.jfif
        │   ├── 230_shetland_sheepdog/
        │   │   └── dogs_init_close_shetland_sheepdog_ILSVRC2012_val_00031875.jfif
        │   ├── 235_german_shepherd_dog/
        │   │   └── dogs_source_german_shepherd_ILSVRC2012_val_00014360.jfif
        │   └── 236_dobermann/
        │       └── dogs2_source_doberman_ILSVRC2012_val_00013033.jfif
        ├── cats/
        │   └── 281_tabby_cat/
        │       └── cats1_source_tabby_281_ILSVRC2012_val_00008876.jfif
        └── cars/
            └── 436_station_wagon/
                └── cars1_source_station_wagon_ILSVRC2012_val_00007374.jfif
```

> **Path Note:** Because `config.json` lives inside `experiments/initialization_study/`, all image paths in `config.json` are written relative to `experiments/initialization_study/` (for example, `images/dogs/236_dobermann/...`).

---

## 2. Configuring `config.json`

Edit `experiments/initialization_study/config.json` to define the experiment groups, baseline settings, and attack parameters.

### Configuration Reference

| Key | Type | Description |
| :--- | :--- | :--- |
| `groups` | `object` | Dictionary of experiment groups (e.g., `"dogs_g1"`, `"cats_g1"`, `"cars_g1"`). |
| `groups.<id>.family` | `string` | Category folder name under `images/` (e.g., `"dogs"`, `"cats"`, `"cars"`). |
| `groups.<id>.source` | `list[string]` | Relative path(s) to the target/source image(s) for that group. |
| `groups.<id>.initializations` | `list[string]` | Relative paths to initialization images (e.g., semantically close and far starting images) used to initialize the attack. |
| `random_baseline` | `boolean` | If `true`, also evaluates a random initialization baseline alongside the specified `initializations`. |
| `attack.name` | `string` | Attack algorithm name (e.g., `"hsja"` for HopSkipJumpAttack). |
| `attack.norms` | `list[string]` | Perturbation norm(s) to evaluate (e.g., `["l2"]`). |
| `attack.seeds` | `list[int]` | Random seed(s) for reproducibility (e.g., `[0]`). |
| `attack.max_queries` | `int` | Maximum total model queries per attack run (e.g., `10000`). |
| `attack.max_unsafe_queries` | `int` | Maximum unsafe/adversarial queries allowed per run (e.g., `10000`). |
| `attack.max_iterations` | `int` | Maximum attack optimization iterations per run (e.g., `500`). |

### Example `experiments/initialization_study/config.json`

```json
{
  "groups": {
    "dogs_g1": {
      "family": "dogs",
      "source": [
        "images/dogs/235_german_shepherd_dog/dogs_source_german_shepherd_ILSVRC2012_val_00014360.jfif"
      ],
      "initializations": [
        "images/dogs/230_shetland_sheepdog/dogs_init_close_shetland_sheepdog_ILSVRC2012_val_00031875.jfif",
        "images/dogs/229_old_english_sheepdog/dogs1_init_far_old_english_sheepdog_ILSVRC2012_val_00030240.jfif"
      ]
    },
    "cats_g1": {
      "family": "cats",
      "source": [
        "images/cats/281_tabby_cat/cats1_source_tabby_281_ILSVRC2012_val_00008876.jfif"
      ],
      "initializations": [
        "images/cats/283_persian_cat/cats1_init_close_persian_ILSVRC2012_val_00002036.jfif",
        "images/cats/290_jaguar/cats1_init_far_jaguar_ILSVRC2012_val_00045488.jfif"
      ]
    }
  },
  "random_baseline": true,
  "attack": {
    "name": "hsja",
    "norms": [
      "l2"
    ],
    "seeds": [
      0
    ],
    "max_queries": 10000,
    "max_unsafe_queries": 10000,
    "max_iterations": 500
  }
}
```

---

## 3. Step 1: Preview Preprocessed Images

Always run the preview job before launching a full attack so you can verify that every image path in `config.json` resolves properly and inspect the preprocessed images prior to attacking.

1. From the repository root directory, submit the preview SLURM job:
   ```bash
   sbatch run_preview.slurm
   ```
2. Monitor the job queue:
   ```bash
   squeue -u $USER
   ```
3. Inspect the generated preview images (in `experiments/initialization_study/previews/`) and check `logs/` to confirm:
   - All `source` and `initializations` image paths load without missing-file errors.
   - Cropping and resizing preserve the target objects properly before running the attack.

---

## 4. Step 2: Run the Full Experiment

Once the previewed images look right, submit the full attack pipeline from the repository root:

1. Launch the experiment via SLURM:
   ```bash
   sbatch run_experiment.slurm
   ```
  
2. Monitor job status and logs:
   ```bash
   squeue -u $USER
   ```
   Check the SLURM logs in `logs/` and experiment outputs under `experiments/initialization_study/results/` as the runs progress.

---

## 5. Step 3: Explore Experiment Results (`summary.csv`)

Once the experiment finishes running, open the generated `summary.csv` file (saved in your experiment output/results directory under `experiments/initialization_study/results/`) to explore detailed information about the results:

- **Attack Outcomes:** Check which individual attack runs succeeded or failed.
- **Perturbation Metrics:** Inspect the final perturbation distance achieved for each run.
- **Detailed Breakdown:** Compare query counts, initialization types, norms, and additional per-run statistics across your configured groups.