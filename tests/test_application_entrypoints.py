"""Real environment lifecycle and subprocess coverage for the benchmark entry."""

import contextlib
from functools import partial
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleoperation.applications import benchmark as application
from teleoperation.simulation.environment import SimulationEnv
from teleoperation.applications import simulation as sessions
from teleoperation.simulation.tasks import get_tasks_by_level


ROOT = Path(__file__).resolve().parents[1]

# Bound the existing run_episode max_steps in the subprocess, while retaining
# the real CLI, reset, randomization, controller, robot loader and physics step.
# Audit actual reset/step calls so a silent no-op or double reset cannot pass.
BOUNDED_CLI = r'''
import json
import sys
import numpy as np
from teleoperation.simulation.environment import SimulationEnv
from teleoperation.applications import simulation as sessions
audit = {"resets": 0, "steps": 0, "tasks": [], "shapes": [], "record_modes": []}
reset, step, episode = SimulationEnv.reset, SimulationEnv.step, sessions.run_episode
def checked_reset(self, *args, **kwargs):
    reset(self, *args, **kwargs)
    audit["resets"] += 1
    audit["tasks"].append(self.task_name)
    audit["record_modes"].append(self.record)
def checked_step(self, action, *args, **kwargs):
    assert self.robot_id is not None and self.action_dim is not None
    assert isinstance(action, np.ndarray) and action.shape == (self.action_dim,)
    assert np.isfinite(action).all()
    audit["steps"] += 1
    audit["shapes"].append(list(action.shape))
    return step(self, action, *args, **kwargs)
def bounded_episode(self, *args, **kwargs):
    kwargs["max_steps"] = 5
    return episode(self, *args, **kwargs)
SimulationEnv.reset, SimulationEnv.step, sessions.run_episode = checked_reset, checked_step, bounded_episode
from teleoperation.applications import benchmark as main
main.run_episode = bounded_episode
sys.argv = ["teleoperation", "sim", "run", *sys.argv[1:]]
from teleoperation.cli import main as cli_main
cli_main()
print("ENTRYPOINT_AUDIT=" + json.dumps(audit))
'''


class ApplicationEntrypointTests(unittest.TestCase):
    def test_controller_reads_live_dimension_after_one_reset_on_each_robot(self):
        with tempfile.TemporaryDirectory() as temp:
            for robot in ("h1_2", "gr1_t2", "g1"):
                with self.subTest(robot=robot), contextlib.redirect_stdout(io.StringIO()):
                    env = sessions.create_environment(robot_type=robot, render=False, record=False, data_dir=temp)
                    try:
                        self.assertIsNone(env.action_dim)
                        self.assertEqual(env.action_joint_names, [])
                        controller = partial(application.demo_controller, env=env)
                        with self.assertRaisesRegex(RuntimeError, r"env.reset\(\)"):
                            controller({})
                        # This is run_episode's normal controller callback, not an
                        # explicit reset added by the entry point.
                        actions = []

                        def checked_controller(obs):
                            action = controller(obs)
                            self.assertIsInstance(action, np.ndarray)
                            self.assertEqual(action.shape, (env.action_dim,))
                            self.assertEqual(env.action_dim, len(env.action_joint_names))
                            self.assertEqual(env.action_dim, len(env.action_joint_indices))
                            self.assertEqual(env.episode_count, 1)
                            self.assertTrue(np.isfinite(action).all())
                            actions.append(action)
                            return action

                        with mock.patch.object(env, "reset", wraps=env.reset) as reset, \
                                mock.patch.object(env.recording.recorder, "reset", wraps=env.recording.recorder.reset) as recorder_reset, \
                                mock.patch.object(env, "step", wraps=env.step) as step:
                            sessions.run_episode(env, controller=checked_controller, max_steps=5, randomize=False)
                        reset.assert_called_once_with(randomize=False)
                        recorder_reset.assert_called_once()
                        self.assertEqual(step.call_count, 5)
                        self.assertEqual(len(actions), 5)
                    finally:
                        env.close()

    def run_cli(self, *args, bounded=True, stdin=None):
        with tempfile.TemporaryDirectory() as temp:
            prefix = ["-c", BOUNDED_CLI] if bounded else ["-m", "teleoperation", "sim", "run"]
            return subprocess.run(
                [sys.executable, "-B", *prefix, *args, "--data-dir", temp],
                cwd=ROOT, input=stdin, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )

    def audit_cli(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("[FAIL]", result.stdout)
        self.assertIn("仿真环境已关闭", result.stdout)
        marker = next(line for line in result.stdout.splitlines() if line.startswith("ENTRYPOINT_AUDIT="))
        return json.loads(marker.split("=", 1)[1])

    def test_demo_alone_executes_three_real_episodes(self):
        result = self.run_cli("--demo", "--no-render", "--no-record")
        audit = self.audit_cli(result)
        self.assertIn("Evaluating: pushcube (3 trials)", result.stdout)
        self.assertEqual(audit["tasks"], ["pushcube"] * 3)
        self.assertEqual(audit["resets"], 3)
        self.assertEqual(audit["steps"], 15)
        self.assertEqual(len(audit["shapes"]), 15)

    def test_demo_default_recording_executes_bounded_real_steps(self):
        result = self.run_cli("--demo", "--no-render")
        audit = self.audit_cli(result)
        self.assertEqual(audit["tasks"], ["pushcube"] * 3)
        self.assertEqual(audit["record_modes"], [True] * 3)
        self.assertEqual(audit["resets"], 3)
        self.assertEqual(audit["steps"], 15)

    def test_explicit_task_selects_demo_task_and_task_alone_runs_one_episode(self):
        for options, trials in ((["--demo", "--task", "pickcube"], 3), (["--task", "pickcube"], 1)):
            with self.subTest(options=options):
                audit = self.audit_cli(self.run_cli(*options, "--no-render", "--no-record"))
                self.assertEqual(audit["tasks"], ["pickcube"] * trials)
                self.assertEqual(audit["resets"], trials)
                self.assertEqual(audit["steps"], 5 * trials)

    def test_benchmark_retains_level_one_task_selection(self):
        audit = self.audit_cli(self.run_cli("--benchmark", "--trials", "1", "--no-render", "--no-record"))
        selected_tasks = get_tasks_by_level(1)
        self.assertEqual(audit["tasks"], selected_tasks)
        self.assertEqual(audit["resets"], len(selected_tasks))
        self.assertEqual(audit["steps"], 5 * len(selected_tasks))

    def test_conflicting_modes_fail_before_environment_creation(self):
        for options in (("--demo", "--benchmark"), ("--task", "pushcube", "--benchmark")):
            with self.subTest(options=options):
                result = self.run_cli(*options, "--no-render", bounded=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("SimulationEnv", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_interactive_demo_and_quit_preserve_existing_routes(self):
        audit = self.audit_cli(self.run_cli("--no-render", "--no-record", stdin="demo\n"))
        self.assertEqual(audit["resets"], 3)
        self.assertEqual(audit["steps"], 15)
        result = self.run_cli("--no-render", "--no-record", stdin="quit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"steps": 0', result.stdout)
        self.assertIn('"resets": 0', result.stdout)


if __name__ == "__main__":
    unittest.main()
