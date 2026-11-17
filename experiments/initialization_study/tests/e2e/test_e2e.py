"""Opt-in real CPU E2E: run this file directly or set RUN_INITIALIZATION_E2E=1."""

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
TIMEOUT_SECONDS = 600
E2E_ENABLED = __name__ == "__main__" or os.environ.get("RUN_INITIALIZATION_E2E") == "1"


@unittest.skipUnless(E2E_ENABLED, "Real E2E is opt-in: run this file directly or set RUN_INITIALIZATION_E2E=1.")
class CPUExperimentE2E(unittest.TestCase):
    """One real image-initialized HSJA task; leaves its artifacts in the normal results directory."""

    """
        Runs the public experiment command with the small fixture and explicit CPU mode.
        Requires a completed attack and real artifacts; skipped tasks do not pass.
        Checks storage and final classification reporting; distance verification remains separate.
    """
    def test_single_source_and_initialization(self) -> None:
        command = [
            sys.executable, "-u", "-m", "experiments.initialization_study.run.run_experiment",
            "--config", str(CONFIG_PATH), "--execute", "--cpu",
        ]
        print("\nRunning real CPU E2E (timeout: {} seconds). Weights may download on first use.".format(
            TIMEOUT_SECONDS,
        ), flush=True)

        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                       timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.fail("CPU E2E exceeded {} seconds. Partial output may remain under results/.".format(TIMEOUT_SECONDS))

        log = completed.stdout + completed.stderr
        print(log, flush=True)
        self.assertEqual(completed.returncode, 0, log)
        output_lines = [line.removeprefix("Saved results: ") for line in completed.stdout.splitlines()
                        if line.startswith("Saved results: ")]
        self.assertEqual(len(output_lines), 1, "Expected exactly one saved experiment directory.\n" + log)
        output_dir = Path(output_lines[0]).resolve()
        self.assertTrue(output_dir.is_relative_to(STUDY_DIR / "results"))
        self.assertTrue(output_dir.is_dir())
        print("E2E artifacts: " + str(output_dir), flush=True)

        snapshot = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot, json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        info = json.loads((output_dir / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["environment"]["device"], "cpu")
        self.assertEqual(info["total_tasks"], 1)
        self.assertEqual(info["finished_tasks"], 1)
        self.assertEqual(info["preparation_queries"], 2)

        with (output_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "completed", "The attack must run, not skip: " + row["reason"])
        self.assertEqual(row["initialization_mode"], "image")
        self.assertEqual(row["source_label_status"], "correct")
        self.assertGreater(int(row["actual_model_queries"]), 0)
        self.assertEqual(row["final_validation_status"], "checked")
        self.assertIn(row["attack_outcome"], ("success", "failure"))
        self.assertEqual(int(row["validation_queries"]), 1)
        expected_success = int(row["final_predicted_class_id"]) != int(row["source_true_class_id"])
        self.assertEqual(row["attack_success"], str(expected_success))
        self.assertTrue(row["final_predicted_class_name"])
        self.assertEqual(info["validation_queries"], 1)
        self.assertEqual(info["verified_tasks"], 1)
        self.assertEqual(info["total_model_queries"], 2 + int(row["actual_model_queries"]) + 1)

        for field in ("result_image", "result_tensor", "task_metadata", "query_history"):
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
        self.assertTrue((task_dir / "initialization.pt").is_file())
        self.assertTrue((output_dir / "data/preflight.json").is_file())
        self.assertTrue((output_dir / "data/input_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
