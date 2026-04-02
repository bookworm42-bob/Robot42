from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any

from .bootstrap import (
    DEFAULT_XLEROBOT_FORK_ROOT,
    XLeRobotBootstrapError,
    XLeRobotBootstrapResult,
    bootstrap_xlerobot,
)


class XLeRobotInterface:
    """Facade for importing and instantiating XLeRobot integrations cleanly."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        force_reload: bool = False,
    ) -> None:
        root = repo_root or os.environ.get("XLEROBOT_FORKED_ROOT") or DEFAULT_XLEROBOT_FORK_ROOT
        self.repo_root = Path(root).expanduser().resolve()
        self.force_reload = force_reload
        self._bootstrap_result: XLeRobotBootstrapResult | None = None

    @property
    def paths(self):
        return self.bootstrap().paths

    def bootstrap(self) -> XLeRobotBootstrapResult:
        if self._bootstrap_result is None or self.force_reload:
            self._bootstrap_result = bootstrap_xlerobot(
                self.repo_root,
                force_reload=self.force_reload,
            )
            self.force_reload = False
        return self._bootstrap_result

    def modules(self) -> dict[str, ModuleType]:
        result = self.bootstrap()
        return {
            "robot": result.robot_module,
            "robot_2wheels": result.robot_2wheels_module,
            "vr": result.vr_module,
            "model": result.model_module,
            "record": result.record_module,
        }

    def robot_classes(self) -> tuple[type[Any], type[Any]]:
        robot_module = self.bootstrap().robot_module
        return robot_module.XLerobotConfig, robot_module.XLerobot

    def robot_2wheels_classes(self) -> tuple[type[Any], type[Any]]:
        module = self.bootstrap().robot_2wheels_module
        return module.XLerobot2WheelsConfig, module.XLerobot2Wheels

    def vr_classes(self) -> tuple[type[Any], type[Any]]:
        vr_module = self.bootstrap().vr_module
        return vr_module.XLerobotVRTeleopConfig, vr_module.XLerobotVRTeleop

    def model_classes(self) -> dict[str, Any]:
        module = self.bootstrap().model_module
        return {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }

    def record_module(self) -> ModuleType:
        return self.bootstrap().record_module

    def make_robot_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.robot_classes()
        return config_cls(**overrides)

    def make_robot(self, **config_overrides: Any) -> Any:
        config_cls, robot_cls = self.robot_classes()
        return robot_cls(config_cls(**config_overrides))

    def make_2wheels_robot_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.robot_2wheels_classes()
        return config_cls(**overrides)

    def make_2wheels_robot(self, **config_overrides: Any) -> Any:
        config_cls, robot_cls = self.robot_2wheels_classes()
        return robot_cls(config_cls(**config_overrides))

    def make_vr_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.vr_classes()
        if "xlevr_path" not in overrides:
            overrides["xlevr_path"] = str(self.paths.xlevr_root)
        return config_cls(**overrides)

    def make_vr_teleop(self, **config_overrides: Any) -> Any:
        config_cls, teleop_cls = self.vr_classes()
        if "xlevr_path" not in config_overrides:
            config_overrides["xlevr_path"] = str(self.paths.xlevr_root)
        return teleop_cls(config_cls(**config_overrides))

    def summary(self) -> dict[str, str]:
        result = self.bootstrap()
        return {
            "repo_root": str(result.paths.repo_root),
            "software_src": str(result.paths.software_src),
            "xlevr_root": str(result.paths.xlevr_root),
            "record_script": str(result.paths.record_script),
            "robot_module": result.robot_module.__name__,
            "robot_2wheels_module": result.robot_2wheels_module.__name__,
            "vr_module": result.vr_module.__name__,
            "model_module": result.model_module.__name__,
            "record_module": result.record_module.__name__,
        }

    @staticmethod
    def installation_help() -> str:
        return (
            "XLeRobot extends LeRobot rather than replacing it. Use a Python "
            "environment where `lerobot` is installed, then point this adapter to "
            "your XLeRobot fork with XLEROBOT_FORKED_ROOT or repo_root=..."
        )
