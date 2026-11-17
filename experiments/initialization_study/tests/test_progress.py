"""Check stage diagnostics and live E2E output without loading a model or running attacks."""

import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from threading import Event

from experiments.initialization_study.progress import stage
from experiments.initialization_study.tests.e2e.test_e2e import run_with_live_output


class ProgressTests(unittest.TestCase):
    def test_stage_reports_start_and_completion(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            with stage('prepare'):
                pass
        self.assertIn('START prepare', output.getvalue())
        self.assertIn('DONE prepare', output.getvalue())

    def test_stage_reports_failure_and_preserves_exception(self) -> None:
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaisesRegex(ValueError, 'broken'):
                with stage('save'):
                    raise ValueError('broken')
        self.assertIn('FAILED save', output.getvalue())
        self.assertIn('ValueError: broken', errors.getvalue())
        self.assertNotIn('DONE save', output.getvalue())

    def test_long_stage_has_heartbeat(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            with stage('waiting', heartbeat_seconds=0.01):
                Event().wait(0.05)
        self.assertIn('WAIT waiting', output.getvalue())

    def test_child_output_and_errors_are_visible_and_retained(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_with_live_output([sys.executable, '-u', '-c',
                'import sys; print("progress"); print("diagnostic", file=sys.stderr)'], 5)
        self.assertEqual(result.returncode, 0)
        self.assertIn('progress', result.stdout)
        self.assertIn('diagnostic', result.stdout)
        self.assertEqual(output.getvalue(), result.stdout)

    def test_silent_child_is_stopped_on_timeout(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired):
            run_with_live_output([sys.executable, '-c', 'import time; time.sleep(30)'], 0.1)
