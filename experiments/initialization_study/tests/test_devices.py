"""Check explicit CPU mode and strict default GPU selection without running ResNet or HSJA."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.integration import model, pipeline
from experiments.initialization_study.planning.planner import build_plan
from experiments.initialization_study.run import run_experiment, run_preflight
from experiments.initialization_study.validation import PlanError
from src.model_wrappers import TorchModelWrapper


class DeviceTests(unittest.TestCase):
    """CPU requires explicit selection; default GPU failures happen before loading weights."""

    def setUp(self) -> None:
        self.config, self.selected = read_config(STUDY_DIR / 'config.json')
        self.jobs, _ = build_plan(self.config, self.selected)

    def test_no_gpu_stops_before_model_or_output_creation(self) -> None:
        with patch.object(model.torch.cuda, 'is_available', return_value=False):
            with patch.object(model, 'resnet50') as factory:
                with patch.object(pipeline, 'RunWriter') as writer:
                    with self.assertRaises(PlanError):
                        pipeline.run_pipeline(self.config, self.selected, self.jobs)
                    with self.assertRaises(PlanError):
                        model.load_target_model('cpu')
                writer.assert_not_called()
            factory.assert_not_called()

    def test_cpu_selection_does_not_probe_cuda(self) -> None:
        with patch.object(model.torch.cuda, 'is_available', side_effect=AssertionError('CUDA probed')):
            self.assertEqual(model.select_device(cpu=True), torch.device('cpu'))

    def test_invalid_gpu_index_stops_before_model_loading(self) -> None:
        with patch.object(model.torch.cuda, 'is_available', return_value=True):
            with patch.object(model.torch.cuda, 'device_count', return_value=1):
                with patch.object(model, 'resnet50') as factory:
                    with self.assertRaises(PlanError):
                        model.load_target_model('cuda:9')
                factory.assert_not_called()

    def test_cpu_loader_keeps_original_wrapper_and_normalization(self) -> None:
        # Replace ResNet construction with an inert module; no forward pass is performed.
        with patch.object(model, 'resnet50', return_value=nn.Identity()) as factory:
            with patch.object(torch.Tensor, 'cuda', side_effect=AssertionError('Unexpected CUDA transfer')):
                wrapper = model.load_target_model(cpu=True)
        self.assertIs(type(wrapper), TorchModelWrapper)
        self.assertEqual(wrapper.im_mean.device.type, 'cpu')
        self.assertEqual(wrapper.im_std.device.type, 'cpu')
        factory.assert_called_once_with(weights=model.ResNet50_Weights.IMAGENET1K_V1)
        pixels = torch.ones((1, 3, 1, 1))
        expected = (pixels - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        expected /= torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(wrapper.preprocess(pixels), expected))

    def test_cpu_task_avoids_cuda_seeding_and_synchronization(self) -> None:
        job = self.jobs[-1]
        source = pipeline.resolve_image(job['source'])
        predictions = {source: {'path': job['source'], 'class_id': 207, 'class_name': 'golden retriever'}}
        target = MagicMock()
        pixels = torch.zeros((1, 3, 224, 224))

        def original_adapter(*args):
            self.assertEqual(args[1].device.type, 'cpu')
            self.assertEqual(args[2].device.type, 'cpu')
            target.num_queries = 2
            return pixels[0], MagicMock(), 0.1, True, {}

        with ExitStack() as mocks:
            mocks.enter_context(patch.object(pipeline, 'load_image_tensor', return_value=pixels))
            mocks.enter_context(patch.object(pipeline, 'run_hsja', side_effect=original_adapter))
            mocks.enter_context(patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CUDA sync')))
            mocks.enter_context(patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA seed')))
            mocks.enter_context(patch.object(torch, 'manual_seed', side_effect=AssertionError('Implicit CUDA seed')))
            result = pipeline.run_task(target, self.config, job, predictions, torch.device('cpu'))

        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.model_queries, 2)

    def test_cpu_and_device_flags_are_mutually_exclusive(self) -> None:
        for command in (run_experiment, run_preflight):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    command.main(['--cpu', '--device', 'cuda:0'])
            self.assertEqual(error.exception.code, 2)

    def test_experiment_cpu_flag_reaches_pipeline(self) -> None:
        with patch.object(run_experiment, 'run_pipeline') as execute:
            execute.return_value.tasks = []
            with redirect_stdout(io.StringIO()):
                run_experiment.main(['--execute', '--cpu'])
            self.assertTrue(execute.call_args.kwargs['cpu'])

    def test_cpu_preflight_report_has_no_gpu_name(self) -> None:
        groups = run_preflight.collect_groups(self.config, self.selected)
        predictions = {}
        for sources, initializations in groups.values():
            for path in sources + initializations:
                predictions[path] = {'path': str(path), 'class_id': 207 if path in sources else 208,
                                     'class_name': 'test class'}
        wrapper = MagicMock()
        wrapper.num_queries = len(predictions)

        with tempfile.TemporaryDirectory() as directory:
            with ExitStack() as mocks:
                mocks.enter_context(patch.object(run_preflight, 'STUDY_DIR', Path(directory)))
                mocks.enter_context(patch.object(run_preflight, 'collect_groups', return_value=groups))
                load = mocks.enter_context(patch.object(run_preflight, 'load_target_model', return_value=wrapper))
                mocks.enter_context(patch.object(run_preflight, 'predict_images', return_value=predictions))
                mocks.enter_context(patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError('GPU probe')))
                with redirect_stdout(io.StringIO()):
                    run_preflight.main(['--cpu', '--config', str(STUDY_DIR / 'config.json')])
                load.assert_called_once_with('cpu', cpu=True)
            report_file = next(Path(directory).glob('preflight_reports/*/report.json'))
            report = json.loads(report_file.read_text())
            self.assertEqual(report['device'], 'cpu')
            self.assertIsNone(report['gpu_name'])


if __name__ == '__main__':
    unittest.main()
