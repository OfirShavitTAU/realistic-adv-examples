"""Real CPU E2E for every group in e2e.json; enabled when this test file is selected."""

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image


TEST_DIR = Path(__file__).resolve().parent
STUDY_DIR = TEST_DIR.parents[1]
PROJECT_ROOT = STUDY_DIR.parents[1]
CONFIG_PATH = TEST_DIR / "e2e.json"
TIMEOUT_SECONDS_PER_TASK = 600
# Enable Play in PyCharm without manual environment setup; an explicit 0 still disables the test.
E2E_ENABLED = os.environ.get("RUN_INITIALIZATION_E2E", "1") == "1"

# Support direct Play independently of PyCharm's working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.initialization_study.configuration.loader import get_source_label, read_config
from experiments.initialization_study.configuration.schemas import StudyConfig
from experiments.initialization_study.planning.planner import build_plan
from experiments.initialization_study.planning.schemas import PlannedJob


"""
    Derives planned tasks and unique preparation inputs from the selected configuration.
    Handles multiple sources, groups, norms, seeds, shared images, and optional random baselines.
"""
def expected_run(config_path: Path) -> tuple[StudyConfig, list[PlannedJob], set[Path]]:
    config, selected = read_config(config_path)
    jobs, _ = build_plan(config, selected)
    paths = {(STUDY_DIR / value).resolve()
             for settings in config["groups"].values()
             for role in ("source", "initializations") for value in settings[role]}

    return config, jobs, paths


@unittest.skipUnless(E2E_ENABLED, "Real E2E is disabled by RUN_INITIALIZATION_E2E; set it to 1 to enable.")
class CPUExperimentE2E(unittest.TestCase):
    """Executes the configured CPU experiment and retains artifacts in the normal results directory."""

    """
        Checks one completed task's exact tensor, display image, query history, and final-label report.
        Random baselines have no external initialization tensor.
    """
    def check_completed_task(self, output_dir: Path, row: dict[str, str], job: PlannedJob) -> None:
        self.assertEqual(row["source_label_status"], "correct")
        self.assertGreater(int(row["actual_model_queries"]), 0)
        self.assertEqual(row["final_validation_status"], "checked")
        self.assertIn(row["attack_outcome"], ("success", "failure"))
        self.assertEqual(int(row["validation_queries"]), 1)
        changed = int(row["final_predicted_class_id"]) != int(row["source_true_class_id"])
        self.assertEqual(row["attack_success"], str(changed))
        self.assertEqual(row["attack_outcome"], "success" if changed else "failure")
        self.assertTrue(row["final_predicted_class_name"])
        self.assertEqual(row["result_image"], job["result_image"])

        for field in ("result_image", "result_tensor", "query_history"):
            self.assertTrue(row[field], "Missing artifact path: " + field)
            self.assertTrue((output_dir / row[field]).is_file(), field)

        pixels = torch.load(output_dir / row["result_tensor"], map_location="cpu", weights_only=True)
        self.assertEqual(tuple(pixels.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(pixels).all().item())

        with Image.open(output_dir / row["result_image"]) as image:
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(image.mode, "RGB")

        history = json.loads((output_dir / row["query_history"]).read_text(encoding="utf-8"))
        self.assertGreater(len(history["history"]), 0)
        task_dir = (output_dir / row["task_metadata"]).parent
        self.assertTrue((task_dir / "source.pt").is_file())
        self.assertEqual((task_dir / "initialization.pt").is_file(), job["initialization_mode"] == "image")

    """
        Verifies saved rows against the whole plan and eligibility against cached preflight predictions.
        Legitimate skips are allowed, but at least one real attack must have completed.
    """
    def check_saved_run(
        self, output_dir: Path, config: StudyConfig, jobs: list[PlannedJob], unique_inputs: set[Path],
    ) -> None:
        snapshot = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot, config)
        saved_plan = json.loads((output_dir / "data/plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_plan, jobs)
        info = json.loads((output_dir / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["environment"]["device"], "cpu")
        self.assertEqual(info["total_tasks"], len(jobs))
        self.assertEqual(info["finished_tasks"], len(jobs))
        self.assertEqual(info["preparation_queries"], len(unique_inputs))
        preflight = json.loads((output_dir / "data/preflight.json").read_text(encoding="utf-8"))
        predictions = {(STUDY_DIR / value["path"]).resolve(): value["class_id"]
                       for value in preflight["predictions"]}
        self.assertEqual(set(predictions), unique_inputs)
        self.assertEqual(len(preflight["predictions"]), len(unique_inputs))
        self.assertTrue((output_dir / "data/input_manifest.json").is_file())

        with (output_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(jobs))
        completed = 0

        for index, (row, job) in enumerate(zip(rows, jobs)):
            with self.subTest(task=index + 1, group=job["group"], source=job["source"],
                              norm=job["norm"], seed=job["seed"], mode=job["initialization_mode"]):
                self.assertEqual(row["task_id"], "task-{:06d}".format(index + 1))
                for field in ("family", "group", "source", "initialization_mode", "norm"):
                    self.assertEqual(row[field], job[field])
                self.assertEqual(row["initialization"], job["initialization"] or "")
                self.assertEqual(int(row["seed"]), job["seed"])
                self.assertTrue(row["task_metadata"])
                metadata = json.loads((output_dir / row["task_metadata"]).read_text(encoding="utf-8"))
                self.assertEqual(metadata["job"], job)

                source_path = (STUDY_DIR / job["source"]).resolve()
                true_class = get_source_label(source_path)
                source_correct = predictions[source_path] == true_class
                eligible = source_correct

                if job["initialization"] is not None:
                    initialization = (STUDY_DIR / job["initialization"]).resolve()
                    eligible = eligible and predictions[initialization] != true_class
                    self.assertEqual(int(row["initialization_predicted_class_id"]), predictions[initialization])

                self.assertEqual(int(row["source_true_class_id"]), true_class)
                self.assertEqual(int(row["source_predicted_class_id"]), predictions[source_path])
                self.assertEqual(row["source_label_status"], "correct" if source_correct else "incorrect")

                if not eligible:
                    self.assertEqual(row["status"], "skipped")
                    self.assertEqual(row["attack_outcome"], "skipped")
                    self.assertTrue(row["reason"])
                    self.assertEqual(int(row["actual_model_queries"]), 0)
                    self.assertEqual(int(row["validation_queries"]), 0)
                    self.assertFalse(row["result_image"])
                    self.assertFalse(row["result_tensor"])
                    continue

                self.assertEqual(row["status"], "completed", row["reason"])
                self.check_completed_task(output_dir, row, job)
                completed += 1

        self.assertGreater(completed, 0, "All tasks were skipped; no attack path was exercised.")
        self.assertEqual(info["verified_tasks"], completed)
        self.assertEqual(info["validation_queries"], completed)
        self.assertEqual(info["actual_attack_model_queries"], sum(int(row["actual_model_queries"]) for row in rows))
        self.assertEqual(info["total_model_queries"], len(unique_inputs) + info["actual_attack_model_queries"] + completed)

        for outcome in ("success", "failure", "skipped", "error", "not_checked"):
            self.assertEqual(info["outcomes"][outcome], sum(row["attack_outcome"] == outcome for row in rows))
        self.assertAlmostEqual(info["verified_success_rate"], info["outcomes"]["success"] / completed)

    """
        Runs every configured task through the public experiment command with explicit CPU mode.
        Timeout scales with task count; output remains available even if assertions fail.
    """
    def test_configured_groups(self) -> None:
        config, jobs, unique_inputs = expected_run(CONFIG_PATH)
        timeout = TIMEOUT_SECONDS_PER_TASK * max(1, len(jobs))
        command = [
            sys.executable, "-u", "-m", "experiments.initialization_study.run.run_experiment",
            "--config", str(CONFIG_PATH), "--cpu",
        ]
        print("\nRunning real CPU E2E: {} tasks, {} unique images (timeout: {} seconds).".format(
            len(jobs), len(unique_inputs), timeout,
        ), flush=True)

        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.fail("CPU E2E exceeded {} seconds. Partial output may remain under results/.".format(timeout))

        log = result.stdout + result.stderr
        print(log, flush=True)
        self.assertEqual(result.returncode, 0, log)
        output_lines = [line.removeprefix("Saved results: ") for line in result.stdout.splitlines()
                        if line.startswith("Saved results: ")]
        self.assertEqual(len(output_lines), 1, "Expected exactly one saved experiment directory.\n" + log)
        output_dir = Path(output_lines[0]).resolve()
        self.assertTrue(output_dir.is_relative_to(STUDY_DIR / "results"))
        self.assertTrue(output_dir.is_dir())
        print("E2E artifacts: " + str(output_dir), flush=True)
        self.check_saved_run(output_dir, config, jobs, unique_inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
