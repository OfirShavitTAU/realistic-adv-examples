"""Validate generalized E2E expectations using synthetic artifacts, without real inference."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.initialization_study.configuration.loader import get_source_label
from experiments.initialization_study.configuration.schemas import StudyConfig
from experiments.initialization_study.planning.schemas import PlannedJob
from experiments.initialization_study.integration.pipeline import TaskResult
from experiments.initialization_study.outputs.writer import RunWriter
from experiments.initialization_study.tests.e2e import test_e2e as e2e
from src.attacks.hsja import HSJAttackPhase
from src.attacks.queries_counter import QueriesCounter


class E2EConfigurationTests(unittest.TestCase):
    """Exercise groups, sources, norms, seeds, baselines, and eligibility-aware skips."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Keep unit fixtures independent of the user's editable real-E2E configuration.
        self.config: StudyConfig = {
            'groups': {
                'first': {
                    'family': 'dogs',
                    'source': ['images/dogs/207_golden_retriever/ILSVRC2012_val_00039893.jfif'],
                    'initializations': [
                        'images/dogs/208_labrador_retriever/ILSVRC2012_val_00043072.jfif',
                        'images/dogs/231_collie/ILSVRC2012_val_00019767.jfif',
                    ],
                },
                'second': {
                    'family': 'dogs',
                    'source': ['images/dogs/231_collie/ILSVRC2012_val_00019767.jfif'],
                    'initializations': ['images/dogs/207_golden_retriever/ILSVRC2012_val_00028946.jfif'],
                },
            },
            'random_baseline': False,
            'attack': {'name': 'hsja', 'norms': ['l2'], 'seeds': [0],
                       'max_queries': 100, 'max_unsafe_queries': 100, 'max_iterations': 1},
        }
        self.checker = e2e.CPUExperimentE2E('test_configured_groups')

    def plan(self, config: StudyConfig) -> tuple[StudyConfig, list[PlannedJob], set[Path]]:
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        return e2e.expected_run(path)

    def saved_run(
        self, config: StudyConfig, invalid_initialization: bool = False, all_sources_wrong: bool = False,
    ) -> tuple[Path, StudyConfig, list[PlannedJob], set[Path]]:
        config, jobs, inputs = self.plan(config)
        groups = {name: ([e2e.STUDY_DIR / path for path in settings['source']],
                         [e2e.STUDY_DIR / path for path in settings['initializations']])
                  for name, settings in config['groups'].items()}
        predictions = {path: {'path': str(path), 'class_id': get_source_label(path), 'class_name': 'mock class'}
                       for path in inputs}
        source_paths = {(e2e.STUDY_DIR / job['source']).resolve() for job in jobs}

        if invalid_initialization:
            job = next(job for job in jobs if job['initialization'] and
                       (e2e.STUDY_DIR / job['initialization']).resolve() not in source_paths)
            predictions[(e2e.STUDY_DIR / job['initialization']).resolve()]['class_id'] = get_source_label(Path(job['source']))

        if all_sources_wrong:
            for path in source_paths:
                predictions[path]['class_id'] = (get_source_label(path) + 1) % 1000

        with patch('experiments.initialization_study.outputs.writer.environment_info', return_value={'device': 'cpu'}):
            writer = RunWriter(config, jobs, 'cpu', self.root / 'result')
        writer.copy_inputs([(settings['family'], name) for name, settings in config['groups'].items()], groups)
        writer.save_preflight(predictions, [], [], len(inputs))
        pixels = torch.zeros((3, 224, 224))
        counter = QueriesCounter(100, 100).increase(
            HSJAttackPhase.gradient_estimation, torch.tensor([True]), torch.tensor([0.1]),
        )

        for index, job in enumerate(jobs):
            source_class = get_source_label(Path(job['source']))
            source_correct = predictions[(e2e.STUDY_DIR / job['source']).resolve()]['class_id'] == source_class
            init_correct = (job['initialization'] is None or
                            predictions[(e2e.STUDY_DIR / job['initialization']).resolve()]['class_id'] != source_class)

            if not source_correct or not init_correct:
                result = TaskResult(job, 'skipped', reason='mock ineligible pair', source_true_class_id=source_class,
                                    source_label_status='correct' if source_correct else 'incorrect')
            else:
                changed = index % 2 == 0
                result = TaskResult(
                    job, 'completed', model_queries=3, output=(pixels, counter, 0.1, True, {}),
                    source_true_class_id=source_class, source_tensor=pixels.unsqueeze(0),
                    initialization_tensor=pixels.unsqueeze(0) if job['initialization'] else None,
                    final_validation_status='checked', final_predicted_class_id=0 if changed else source_class,
                    final_predicted_class_name='mock final class', attack_success=changed, validation_queries=1,
                )
            writer.save_task(index, result)
        writer.finish('completed')
        return writer.directory, config, jobs, inputs

    def test_plan_scales_with_sources_norms_seeds_and_random_baselines(self) -> None:
        config = copy.deepcopy(self.config)
        config['random_baseline'] = True
        config['attack']['norms'] = ['l2', 'linf']
        config['attack']['seeds'] = [0, 1]
        first_group = next(iter(config['groups'].values()))
        first_group['source'].append('images/dogs/207_golden_retriever/ILSVRC2012_val_00028946.jfif')
        config, jobs, inputs = self.plan(config)
        expected = sum(len(group['source']) * (len(group['initializations']) + 1) * 4
                       for group in config['groups'].values())
        self.assertEqual(len(jobs), expected)
        self.assertEqual(len(inputs), len({(e2e.STUDY_DIR / value).resolve()
                                          for group in config['groups'].values()
                                          for role in ('source', 'initializations') for value in group[role]}))

    def test_multiple_groups_with_random_baselines_pass_artifact_checks(self) -> None:
        config = copy.deepcopy(self.config)
        config['random_baseline'] = True
        self.checker.check_saved_run(*self.saved_run(config))

    def test_justified_skips_are_accepted_alongside_completed_attacks(self) -> None:
        self.checker.check_saved_run(*self.saved_run(self.config, invalid_initialization=True))

    def test_all_skipped_does_not_count_as_end_to_end_execution(self) -> None:
        with self.assertRaisesRegex(AssertionError, 'All tasks were skipped'):
            self.checker.check_saved_run(*self.saved_run(self.config, all_sources_wrong=True))

    def test_mismatched_group_or_task_identity_is_rejected(self) -> None:
        directory, config, jobs, inputs = self.saved_run(self.config)
        changed_jobs = copy.deepcopy(jobs)
        changed_jobs[0]['group'] = 'wrong_group'
        with self.assertRaises(AssertionError):
            self.checker.check_saved_run(directory, config, changed_jobs, inputs)


if __name__ == '__main__':
    unittest.main()
