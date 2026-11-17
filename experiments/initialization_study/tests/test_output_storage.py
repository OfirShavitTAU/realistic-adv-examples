"""Exercise real artifact storage with synthetic CPU tensors and mocked execution only."""

import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.integration import pipeline
from experiments.initialization_study.integration.pipeline import TaskResult
from experiments.initialization_study.outputs.serialization import write_json
from experiments.initialization_study.outputs.writer import RunWriter
from experiments.initialization_study.planning.planner import build_plan
from experiments.initialization_study.run.run_preflight import collect_groups
from src.attacks.hsja import HSJAttackPhase
from src.attacks.queries_counter import QueriesCounter


class OutputStorageTests(unittest.TestCase):
    """Verify exact artifacts, incremental state, and failures without real inference or attacks."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config, self.selected = read_config(STUDY_DIR / "config.json")
        self.jobs, _ = build_plan(self.config, self.selected)
        self.groups = collect_groups(self.config, self.selected)
        self.predictions = {}

        for job in self.jobs:
            self.predictions[pipeline.resolve_image(job["source"])] = {
                "path": job["source"], "class_id": 207, "class_name": "golden retriever",
            }

            if job["initialization"]:
                self.predictions[pipeline.resolve_image(job["initialization"])] = {
                    "path": job["initialization"], "class_id": 208, "class_name": "Labrador retriever",
                }

        metadata = patch('experiments.initialization_study.outputs.writer.environment_info',
                         return_value={"device": "cuda:0", "git": {"commit": "test", "dirty": False}})
        metadata.start()
        self.addCleanup(metadata.stop)

    def result(self, index: int = 0) -> TaskResult:
        pixels = torch.linspace(0.001, 0.999, 3 * 224 * 224, dtype=torch.float32).reshape(3, 224, 224)
        counter = QueriesCounter(2000, 1000).increase(
            HSJAttackPhase.gradient_estimation, torch.tensor([True, False]), torch.tensor([0.2, 0.3]),
        )
        return TaskResult(self.jobs[index], "completed", 0.25, 3, (pixels, counter, 0.2, True, {}),
                          source_true_class_id=207, source_tensor=pixels.unsqueeze(0),
                          initialization_tensor=pixels.unsqueeze(0) if self.jobs[index]["initialization"] else None)

    def writer(self) -> RunWriter:
        writer = RunWriter(self.config, self.jobs, 'cuda:0', self.root / 'experiment')
        writer.copy_inputs(self.selected, self.groups)
        writer.save_preflight(self.predictions, [], [], 3)
        return writer

    def rows(self, directory: Path) -> list[dict[str, str]]:
        with (directory / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_exact_tensors_display_tree_and_query_history(self) -> None:
        writer = self.writer()
        outcome = self.result()
        writer.start_task(0)
        writer.save_task(0, outcome)
        rows = self.rows(writer.directory)
        row = rows[0]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(rows[1]['status'], 'pending')
        self.assertEqual(row['actual_model_queries'], '3')
        self.assertEqual(row['reported_total_queries'], '2')
        self.assertEqual(row['reported_safe_queries'], '1')
        self.assertEqual(row['reported_unsafe_queries'], '1')
        self.assertEqual(row['final_validation_status'], 'not_checked')
        self.assertEqual(row['l2'], '')
        self.assertEqual(row['linf'], '')
        self.assertEqual(row['attack_reported_success'], 'True')
        self.assertTrue(row['result_image'].endswith('-atttack.png'))
        saved = torch.load(writer.directory / row['result_tensor'], map_location='cpu', weights_only=True)
        self.assertTrue(torch.equal(saved, outcome.output[0]))
        source = torch.load(writer.directory / 'data/task-000001/source.pt', weights_only=True)
        self.assertTrue(torch.equal(source, outcome.source_tensor))
        self.assertTrue((writer.directory / 'data/task-000001/initialization.pt').exists())

        with Image.open(writer.directory / row['result_image']) as image:
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(image.mode, 'RGB')

        history = json.loads((writer.directory / row['query_history']).read_text())
        self.assertEqual(len(history['history']), 2)
        self.assertEqual(history['history'][0]['phase'], 'gradient_estimation')
        self.assertTrue(history['history'][0]['safe'])
        self.assertFalse(history['history'][1]['safe'])
        manifest = json.loads((writer.directory / 'data/input_manifest.json').read_text())
        self.assertEqual(len(manifest), 3)

        for entry in manifest:
            self.assertEqual(Path(entry['original_path']).read_bytes(),
                             (writer.directory / entry['copied_path']).read_bytes())

        self.assertEqual(json.loads((writer.directory / 'config.json').read_text()), self.config)
        info = json.loads((writer.directory / 'run_info.json').read_text())
        self.assertEqual(info['preparation_queries'], 3)
        self.assertEqual(info['finished_tasks'], 1)
        self.assertIn('l2', info['effective_attack_parameters'])
        self.assertEqual(info['effective_attack_parameters']['l2']['queries_limit'], 2000)

    def test_skipped_and_failed_tasks_have_metadata_without_fake_outputs(self) -> None:
        writer = self.writer()
        writer.save_task(0, TaskResult(self.jobs[0], 'skipped', reason='same class', source_true_class_id=207))
        writer.save_task(1, TaskResult(self.jobs[1], 'error', model_queries=9, reason='synthetic failure',
                                      source_true_class_id=207))
        rows = self.rows(writer.directory)

        for row, expected in zip(rows, ('skipped', 'error')):
            self.assertEqual(row['status'], expected)
            self.assertEqual(row['result_tensor'], '')
            self.assertEqual(row['result_image'], '')
            metadata = json.loads((writer.directory / row['task_metadata']).read_text())
            self.assertEqual(metadata['query_history_status'], 'unavailable')
            self.assertEqual(metadata['final_validation_status'], 'not_checked')

        info = json.loads((writer.directory / 'run_info.json').read_text())
        self.assertEqual(info['actual_attack_model_queries'], 9)

    def test_random_baseline_uses_reserved_filename_without_initialization_tensor(self) -> None:
        writer = self.writer()
        writer.save_task(2, self.result(2))
        row = self.rows(writer.directory)[2]
        self.assertTrue(row['result_image'].endswith('/random-atttack.png'))
        self.assertTrue((writer.directory / row['result_image']).exists())
        self.assertFalse((writer.directory / 'data/task-000003/initialization.pt').exists())

    def test_verified_outcomes_names_and_success_rate_are_saved(self) -> None:
        writer = self.writer()
        for index, success in enumerate((True, False)):
            result = self.result(index)
            result.final_validation_status = 'checked'
            result.final_predicted_class_id = 208 if success else 207
            result.final_predicted_class_name = 'Labrador retriever' if success else 'golden retriever'
            result.attack_success = success
            result.validation_queries = 1
            writer.save_task(index, result)
        writer.save_task(2, TaskResult(self.jobs[2], 'skipped', reason='test skip'))
        rows = self.rows(writer.directory)
        self.assertEqual([row['attack_outcome'] for row in rows], ['success', 'failure', 'skipped'])
        self.assertEqual(rows[1]['attack_reported_success'], 'True')
        self.assertEqual(rows[1]['attack_success'], 'False')
        self.assertEqual(rows[0]['final_predicted_class_name'], 'Labrador retriever')
        self.assertEqual(rows[0]['total_task_model_queries'], '4')
        self.assertEqual(rows[0]['distance_validation_status'], 'not_checked')
        info = json.loads((writer.directory / 'run_info.json').read_text())
        self.assertEqual(info['outcomes']['success'], 1)
        self.assertEqual(info['outcomes']['failure'], 1)
        self.assertEqual(info['outcomes']['skipped'], 1)
        self.assertEqual(info['verified_success_rate'], 0.5)
        self.assertEqual(info['verified_tasks'], 2)
        self.assertEqual(info['validation_queries'], 2)
        self.assertEqual(info['total_model_queries'], 11)
        self.assertEqual(info['task_ids_by_outcome']['failure'], ['task-000002'])

    def test_validation_errors_keep_original_output_and_are_not_counted_as_failures(self) -> None:
        writer = self.writer()
        result = self.result()
        result.status = 'error'
        result.final_validation_status = 'error'
        result.validation_error = 'synthetic verification error'
        result.validation_queries = 1
        writer.save_task(0, result)
        row = self.rows(writer.directory)[0]
        self.assertEqual(row['attack_outcome'], 'error')
        self.assertEqual(row['attack_success'], '')
        self.assertTrue((writer.directory / row['result_tensor']).exists())
        info = json.loads((writer.directory / 'run_info.json').read_text())
        self.assertEqual(info['outcomes']['error'], 1)
        self.assertEqual(info['outcomes']['failure'], 0)
        self.assertIsNone(info['verified_success_rate'])
        self.assertEqual(info['final_validation_status'], 'error')

    def test_existing_run_directory_not_overwritten(self) -> None:
        writer = self.writer()
        snapshot = (writer.directory / 'config.json').read_bytes()
        with self.assertRaises(FileExistsError):
            RunWriter(self.config, self.jobs, 'cuda:0', writer.directory)
        self.assertEqual((writer.directory / 'config.json').read_bytes(), snapshot)

    def test_failed_artifact_write_does_not_publish_a_completed_row(self) -> None:
        writer = self.writer()
        writer.start_task(0)
        with patch('experiments.initialization_study.outputs.writer.write_image', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                writer.save_task(0, self.result())
        writer.finish('failed', 'disk full')
        self.assertEqual(self.rows(writer.directory)[0]['status'], 'running')
        self.assertEqual(json.loads((writer.directory / 'run_info.json').read_text())['status'], 'failed')

    def test_atomic_json_preserves_old_file_on_serialization_failure(self) -> None:
        path = self.root / 'metadata.json'
        write_json(path, {'value': 1})
        with self.assertRaises(TypeError):
            write_json(path, {'value': object()})
        self.assertEqual(json.loads(path.read_text()), {'value': 1})
        write_json(path, {'value': float('inf')})
        self.assertIsNone(json.loads(path.read_text())['value'])

    def test_pipeline_interruption_keeps_completed_task_and_marks_active_task(self) -> None:
        model = MagicMock()
        model.num_queries = 3
        output_dir = self.root / 'interrupted-run'

        with ExitStack() as mocks:
            mocks.enter_context(patch.object(pipeline, 'select_device', return_value=torch.device('cuda:0')))
            mocks.enter_context(patch.object(pipeline, 'collect_groups', return_value=self.groups))
            mocks.enter_context(patch.object(pipeline, 'load_target_model', return_value=model))
            mocks.enter_context(patch.object(pipeline, 'predict_images', return_value=self.predictions))
            mocks.enter_context(patch.object(pipeline, 'run_task', side_effect=[self.result(), KeyboardInterrupt()]))
            with self.assertRaises(KeyboardInterrupt):
                pipeline.run_pipeline(self.config, self.selected, self.jobs, output_dir=output_dir)

        rows = self.rows(output_dir)
        self.assertEqual([row['status'] for row in rows], ['completed', 'running', 'pending'])
        self.assertTrue((output_dir / rows[0]['result_tensor']).exists())
        info = json.loads((output_dir / 'run_info.json').read_text())
        self.assertEqual(info['status'], 'interrupted')
        self.assertEqual(info['active_task_id'], 'task-000002')
        self.assertEqual(info['finished_tasks'], 1)

    def test_setup_failure_preserves_partial_preparation_count(self) -> None:
        model = MagicMock()
        model.num_queries = 2
        output_dir = self.root / 'failed-preparation'

        with ExitStack() as mocks:
            mocks.enter_context(patch.object(pipeline, 'select_device', return_value=torch.device('cuda:0')))
            mocks.enter_context(patch.object(pipeline, 'collect_groups', return_value=self.groups))
            mocks.enter_context(patch.object(pipeline, 'load_target_model', return_value=model))
            mocks.enter_context(patch.object(pipeline, 'predict_images', side_effect=RuntimeError('prediction failed')))
            with self.assertRaises(RuntimeError):
                pipeline.run_pipeline(self.config, self.selected, self.jobs, output_dir=output_dir)

        info = json.loads((output_dir / 'run_info.json').read_text())
        self.assertEqual(info['status'], 'failed')
        self.assertEqual(info['preparation_queries'], 2)
        self.assertEqual([row['status'] for row in self.rows(output_dir)], ['pending'] * 3)

    def test_pipeline_saves_all_outcomes_and_finishes_with_errors(self) -> None:
        model = MagicMock()
        model.num_queries = 3
        output_dir = self.root / 'completed-run'
        outcomes = [self.result(), TaskResult(self.jobs[1], 'skipped', reason='same class'),
                    TaskResult(self.jobs[2], 'error', model_queries=7, reason='synthetic failure')]

        with ExitStack() as mocks:
            mocks.enter_context(patch.object(pipeline, 'select_device', return_value=torch.device('cuda:0')))
            mocks.enter_context(patch.object(pipeline, 'collect_groups', return_value=self.groups))
            mocks.enter_context(patch.object(pipeline, 'load_target_model', return_value=model))
            mocks.enter_context(patch.object(pipeline, 'predict_images', return_value=self.predictions))
            mocks.enter_context(patch.object(pipeline, 'run_task', side_effect=outcomes))
            result = pipeline.run_pipeline(self.config, self.selected, self.jobs, output_dir=output_dir)

        self.assertEqual(result.output_dir, output_dir)
        info = json.loads((output_dir / 'run_info.json').read_text())
        self.assertEqual(info['status'], 'completed_with_errors')
        self.assertEqual(info['actual_attack_model_queries'], 10)
        self.assertEqual(info['preparation_queries'], 3)
        self.assertEqual([row['status'] for row in self.rows(output_dir)], ['completed', 'skipped', 'error'])


if __name__ == '__main__':
    unittest.main()
