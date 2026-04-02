"""Clean integration layer for consuming XLeRobot from this repository.

This package does not copy XLeRobot code. It bootstraps a forked XLeRobot repo
into the installed `lerobot` namespace and exposes a stable facade on top.
"""

from .bootstrap import XLeRobotBootstrapError, XLeRobotBootstrapResult, bootstrap_xlerobot
from .interface import XLeRobotInterface
from .types import XLeRobotPaths

__all__ = [
    "XLeRobotBootstrapError",
    "XLeRobotBootstrapResult",
    "XLeRobotInterface",
    "XLeRobotPaths",
    "bootstrap_xlerobot",
    "XLeRobotManiSkillBootstrapResult",
    "XLeRobotManiSkillError",
    "bootstrap_xlerobot_maniskill",
    "run_keyboard_play_demo",
]


def __getattr__(name: str):
    if name in {
        "XLeRobotManiSkillBootstrapResult",
        "XLeRobotManiSkillError",
        "bootstrap_xlerobot_maniskill",
        "run_keyboard_play_demo",
    }:
        from .maniskill import (
            XLeRobotManiSkillBootstrapResult,
            XLeRobotManiSkillError,
            bootstrap_xlerobot_maniskill,
            run_keyboard_play_demo,
        )

        exports = {
            "XLeRobotManiSkillBootstrapResult": XLeRobotManiSkillBootstrapResult,
            "XLeRobotManiSkillError": XLeRobotManiSkillError,
            "bootstrap_xlerobot_maniskill": bootstrap_xlerobot_maniskill,
            "run_keyboard_play_demo": run_keyboard_play_demo,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
