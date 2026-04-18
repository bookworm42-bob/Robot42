import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from multido_xlerobot.bootstrap import _default_repo_root, resolve_xlerobot_repo_root
from multido_xlerobot.interface import _patch_imported_vr_monitor_path
from multido_xlerobot.maniskill import _prepare_vr_monitor_module


class XLeRobotPathTests(unittest.TestCase):
    def test_default_prefers_forked_checkout_over_upstream_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            forked = home / "xlerobot_forked"
            upstream = home / "XLeRobot"
            forked.mkdir()
            upstream.mkdir()

            with patch("multido_xlerobot.bootstrap.Path.home", return_value=home):
                self.assertEqual(_default_repo_root(), forked)

    def test_env_override_is_used_when_no_explicit_root_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "custom_xlerobot"
            with patch.dict(os.environ, {"XLEROBOT_FORKED_ROOT": str(env_root)}):
                self.assertEqual(resolve_xlerobot_repo_root(), env_root.resolve())

    def test_explicit_root_wins_over_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            explicit_root = Path(tmp) / "explicit"
            env_root = Path(tmp) / "env"
            with patch.dict(os.environ, {"XLEROBOT_FORKED_ROOT": str(env_root)}):
                self.assertEqual(resolve_xlerobot_repo_root(explicit_root), explicit_root.resolve())

    def test_vr_monitor_path_patch_updates_imported_module(self) -> None:
        module_name = "lerobot.teleoperators.xlerobot_vr.vr_monitor"
        monitor_module = types.ModuleType(module_name)
        monitor_module.XLEVR_PATH = "/stale/path"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {module_name: monitor_module}):
            xlevr_root = Path(tmp) / "XLeVR"
            self.assertTrue(
                _patch_imported_vr_monitor_path(
                    "lerobot.teleoperators.xlerobot_vr",
                    xlevr_root,
                )
            )
            self.assertEqual(monitor_module.XLEVR_PATH, str(xlevr_root.resolve()))

    def test_sim_vr_monitor_is_loaded_from_forked_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            monitor_file = repo_root / "simulation" / "Maniskill" / "examples" / "vr_monitor.py"
            monitor_file.parent.mkdir(parents=True)
            monitor_file.write_text("XLEVR_PATH = '/stale/path'\n")
            (repo_root / "XLeVR").mkdir()
            mani_skill_module = types.ModuleType("mani_skill")
            with patch.dict(sys.modules, {"mani_skill": mani_skill_module}):
                module = _prepare_vr_monitor_module(repo_root, force_reload=True)
                self.assertEqual(module.XLEVR_PATH, str((repo_root / "XLeVR").resolve()))
                self.assertIs(sys.modules["mani_skill.examples"].vr_monitor, module)


if __name__ == "__main__":
    unittest.main()
