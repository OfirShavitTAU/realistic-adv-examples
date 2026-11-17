"""Verify orchestration with mocks; no GPU inference or real attacks are executed."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.integration import pipeline
from experiments.initialization_study.integration.verification import FinalValidation
from experiments.initialization_study.planning.planner import build_plan


class PipelineTests(unittest.TestCase):
    """Cover preparation, task isolation, skipping, seeds, and error continuation."""

    def setUp(self) -> None:
        writer_patch = patch.object(pipeline, "RunWriter")
        self.writer = writer_patch.start().return_value
        self.addCleanup(writer_patch.stop)
        validation_patch = patch.object(pipeline, "verify_final_prediction",
                                        return_value=FinalValidation("checked", 208, "Labrador retriever", True, 1))
        self.verifier = validation_patch.start()
        self.addCleanup(validation_patch.stop)
        self.config, self.selected = read_config(STUDY_DIR / "config.json")
        self.jobs, _ = build_plan(self.config, self.selected)
        self.device = torch.device("cuda:0")
        self.groups = {
            group: (
                [pipeline.resolve_image(path) for path in settings["source"]],
                [pipeline.resolve_image(path) for path in settings["initializations"]],
            )
            for group, settings in self.config["groups"].items()
        }
        self.predictions = {}

        for job in self.jobs:
            self.predictions[pipeline.resolve_image(job["source"])] = {
                "path": job["source"], "class_id": 207, "class_name": "golden retriever",
            }

            if job["initialization"] is not None:
                self.predictions[pipeline.resolve_image(job["initialization"])] = {
                    "path": job["initialization"], "class_id": 208, "class_name": "Labrador retriever",
                }

    def test_model_loaded_once_and_preparation_kept_separate(self) -> None:
        model = MagicMock()
        model.num_queries = 3
        outcomes = [pipeline.TaskResult(job, "completed", model_queries=10) for job in self.jobs]

        with patch.object(pipeline, "select_device", return_value=self.device):
            with patch.object(pipeline, "collect_groups", return_value=self.groups):
                with patch.object(pipeline, "load_target_model", return_value=model) as load:
                    with patch.object(pipeline, "predict_images", return_value=self.predictions) as predict:
                        with patch.object(pipeline, "check_pairs"):
                            with patch.object(pipeline, "run_task", side_effect=outcomes) as run:
                                result = pipeline.run_pipeline(self.config, self.selected, self.jobs)

        load.assert_called_once_with("cuda:0", cpu=False)
        predict.assert_called_once()
        self.assertEqual(result.preparation_queries, 3)
        self.assertEqual(result.tasks, outcomes)
        self.assertEqual(run.call_count, len(self.jobs))
        self.assertTrue(all(call.args[0] is model for call in run.call_args_list))

    def test_invalid_initialization_skipped_without_queries(self) -> None:
        job = self.jobs[0]
        path = pipeline.resolve_image(job["initialization"])
        self.predictions[path]["class_id"] = 207

        with patch.object(pipeline, "run_hsja") as attack:
            result = pipeline.run_task(MagicMock(), self.config, job, self.predictions, self.device)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.model_queries, 0)
        attack.assert_not_called()

    def test_seed_and_original_adapter_for_both_initialization_modes(self) -> None:
        for job in (self.jobs[0], self.jobs[-1]):
            model = MagicMock()
            image = MagicMock()
            output_image = MagicMock()
            counter = MagicMock()

            def attack(*args):
                self.assertEqual(model.num_queries, 0)
                model.num_queries += 17
                return output_image, counter, 0.5, True, {}

            with patch.object(pipeline, "load_image_tensor", return_value=image) as prepare:
                with patch.object(pipeline.torch, "tensor", return_value=MagicMock()):
                    with patch.object(pipeline.torch.cuda, "synchronize"):
                        with patch.object(pipeline, "set_task_seed") as seed:
                            with patch.object(pipeline, "run_hsja", side_effect=attack) as adapter:
                                result = pipeline.run_task(model, self.config, job, self.predictions, self.device)

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.model_queries, 17)
            self.assertEqual(result.validation_queries, 1)
            self.assertEqual(result.final_predicted_class_id, 208)
            self.assertEqual(result.outcome, "success")
            self.assertIs(result.output[0], output_image.detach.return_value.cpu.return_value)
            seed.assert_called_once_with(job["seed"], self.device)
            self.assertEqual(adapter.call_args.args[4], job["norm"])
            self.assertEqual(prepare.call_count, 1 if job["initialization"] is None else 2)

            if job["initialization"] is None:
                self.assertIsNone(adapter.call_args.args[5])

    def test_failed_prediction_and_validation_errors_preserve_original_outputs(self) -> None:
        validations = (FinalValidation("checked", 207, "golden retriever", False, 1),
                       FinalValidation("error", queries=1, error="synthetic validation error"))
        pixels = torch.zeros(1, 3, 224, 224)

        for validation in validations:
            model = MagicMock()

            def attack(*args):
                model.num_queries = 5
                return pixels[0], MagicMock(), 0.5, True, {}

            def verify(*args):
                model.num_queries += 1
                return validation

            self.verifier.side_effect = verify

            with patch.object(pipeline, "load_image_tensor", return_value=pixels):
                with patch.object(pipeline, "run_hsja", side_effect=attack):
                    with patch.object(pipeline, "set_task_seed"):
                        result = pipeline.run_task(model, self.config, self.jobs[-1], self.predictions,
                                                   torch.device("cpu"))

            self.assertIsNotNone(result.output)
            self.assertTrue(result.output[3])
            self.assertEqual(result.model_queries, 5)
            self.assertEqual(result.validation_queries, 1)
            self.assertEqual(model.num_queries, 6)
            self.assertEqual(result.outcome, "failure" if validation.status == "checked" else "error")
            self.assertEqual(result.attack_success, validation.attack_success)

    def test_task_failure_retains_consumed_queries(self) -> None:
        model = MagicMock()

        def fail(*args):
            model.num_queries = 9
            raise RuntimeError("test failure")

        with patch.object(pipeline, "load_image_tensor", return_value=MagicMock()):
            with patch.object(pipeline.torch, "tensor", return_value=MagicMock()):
                with patch.object(pipeline.torch.cuda, "synchronize"):
                    with patch.object(pipeline, "set_task_seed"):
                        with patch.object(pipeline, "run_hsja", side_effect=fail):
                            result = pipeline.run_task(model, self.config, self.jobs[-1], self.predictions, self.device)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.model_queries, 9)
        self.assertIn("test failure", result.reason)

    def test_pipeline_continues_after_task_error(self) -> None:
        model = MagicMock()
        model.num_queries = 3
        outcomes = [pipeline.TaskResult(self.jobs[0], "error", reason="test error")]
        outcomes += [pipeline.TaskResult(job, "completed") for job in self.jobs[1:]]

        with patch.object(pipeline, "select_device", return_value=self.device):
            with patch.object(pipeline, "collect_groups", return_value=self.groups):
                with patch.object(pipeline, "load_target_model", return_value=model):
                    with patch.object(pipeline, "predict_images", return_value=self.predictions):
                        with patch.object(pipeline, "check_pairs"):
                            with patch.object(pipeline, "run_task", side_effect=outcomes):
                                result = pipeline.run_pipeline(self.config, self.selected, self.jobs)

        self.assertEqual([task.status for task in result.tasks], ["error", "completed", "completed"])

    def test_misclassified_source_skips_image_and_random_tasks(self) -> None:
        source = pipeline.resolve_image(self.jobs[0]["source"])
        self.predictions[source]["class_id"] = 231

        with patch.object(pipeline, "run_hsja") as attack:
            with patch.object(pipeline, "load_image_tensor") as prepare:
                for job in self.jobs:
                    result = pipeline.run_task(MagicMock(), self.config, job, self.predictions, self.device)
                    self.assertEqual(result.status, "skipped")
                    self.assertEqual(result.source_label_status, "incorrect")
                    self.assertEqual(result.source_true_class_id, 207)
                    self.assertEqual(result.model_queries, 0)
                attack.assert_not_called()
                prepare.assert_not_called()

    def test_generators_seeded_per_task(self) -> None:
        with patch.object(pipeline.random, "seed") as python_seed:
            with patch.object(pipeline.np.random, "seed") as numpy_seed:
                with patch.object(pipeline.torch, "manual_seed") as torch_seed:
                    with patch.object(pipeline.torch.cuda, "manual_seed_all") as cuda_seed:
                        pipeline.set_task_seed(42, self.device)
        for setter in (python_seed, numpy_seed, torch_seed, cuda_seed):
            setter.assert_called_once_with(42)


if __name__ == "__main__":
    unittest.main()
