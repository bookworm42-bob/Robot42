from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .types import XLeRobotPaths


DEFAULT_XLEROBOT_FORK_ROOT = Path("/Users/alin/xlerobot_forked")


class XLeRobotBootstrapError(RuntimeError):
    """Raised when the local XLeRobot fork cannot be exposed cleanly."""


@dataclass(frozen=True)
class XLeRobotBootstrapResult:
    paths: XLeRobotPaths
    robot_module: ModuleType
    robot_2wheels_module: ModuleType
    vr_module: ModuleType
    model_module: ModuleType
    record_module: ModuleType


def bootstrap_xlerobot(
    repo_root: str | Path = DEFAULT_XLEROBOT_FORK_ROOT,
    *,
    force_reload: bool = False,
) -> XLeRobotBootstrapResult:
    paths = XLeRobotPaths.from_repo_root(repo_root)
    _validate_paths(paths)
    _ensure_lerobot_installed()

    model_module = _load_module(
        "lerobot.model.SO101Robot",
        paths.model_file,
        force_reload=force_reload,
    )
    robot_module = _load_package(
        "lerobot.robots.xlerobot",
        paths.robot_pkg_dir,
        force_reload=force_reload,
    )
    robot_2wheels_module = _load_package(
        "lerobot.robots.xlerobot_2wheels",
        paths.robot_2wheels_pkg_dir,
        force_reload=force_reload,
    )
    vr_module = _load_package(
        "lerobot.teleoperators.xlerobot_vr",
        paths.vr_pkg_dir,
        force_reload=force_reload,
    )
    record_module = _load_module(
        "xlerobot_forked.record",
        paths.record_script,
        force_reload=force_reload,
    )

    return XLeRobotBootstrapResult(
        paths=paths,
        robot_module=robot_module,
        robot_2wheels_module=robot_2wheels_module,
        vr_module=vr_module,
        model_module=model_module,
        record_module=record_module,
    )


def _validate_paths(paths: XLeRobotPaths) -> None:
    required = {
        "repo_root": paths.repo_root,
        "software_src": paths.software_src,
        "xlevr_root": paths.xlevr_root,
        "record_script": paths.record_script,
        "model_file": paths.model_file,
        "robot_pkg_dir": paths.robot_pkg_dir,
        "robot_2wheels_pkg_dir": paths.robot_2wheels_pkg_dir,
        "vr_pkg_dir": paths.vr_pkg_dir,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        formatted = ", ".join(f"{name}={required[name]}" for name in missing)
        raise XLeRobotBootstrapError(
            f"XLeRobot fork is incomplete or moved. Missing: {formatted}"
        )


def _ensure_lerobot_installed() -> None:
    try:
        importlib.import_module("lerobot")
    except ModuleNotFoundError as exc:
        raise XLeRobotBootstrapError(
            "The XLeRobot fork extends the `lerobot` package, but `lerobot` is not "
            "installed in the current Python environment. Install a compatible "
            "LeRobot environment first, then bootstrap XLeRobot."
        ) from exc


def _load_package(module_name: str, package_dir: Path, *, force_reload: bool) -> ModuleType:
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise XLeRobotBootstrapError(f"Missing package entrypoint: {init_file}")

    if force_reload:
        _purge_modules(module_name)
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise XLeRobotBootstrapError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module(module_name: str, file_path: Path, *, force_reload: bool) -> ModuleType:
    if not file_path.exists():
        raise XLeRobotBootstrapError(f"Missing module file: {file_path}")

    if force_reload and module_name in sys.modules:
        del sys.modules[module_name]
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise XLeRobotBootstrapError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _purge_modules(prefix: str) -> None:
    names = [name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")]
    for name in names:
        del sys.modules[name]
