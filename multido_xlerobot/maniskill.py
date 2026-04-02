from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .bootstrap import DEFAULT_XLEROBOT_FORK_ROOT


class XLeRobotManiSkillError(RuntimeError):
    """Raised when the local ManiSkill + XLeRobot play stack is not ready."""


@dataclass(frozen=True)
class XLeRobotManiSkillBootstrapResult:
    repo_root: Path
    sim_root: Path
    assets_root: Path
    agent_module: ModuleType
    env_module: ModuleType


def bootstrap_xlerobot_maniskill(
    repo_root: str | Path = DEFAULT_XLEROBOT_FORK_ROOT,
    *,
    force_reload: bool = False,
) -> XLeRobotManiSkillBootstrapResult:
    root = Path(repo_root).expanduser().resolve()
    sim_root = root / "simulation" / "Maniskill"
    assets_root = sim_root / "assets" / "xlerobot"
    agent_file = sim_root / "agents" / "xlerobot" / "xlerobot.py"
    env_file = sim_root / "envs" / "scenes" / "base_env.py"

    required = {
        "repo_root": root,
        "sim_root": sim_root,
        "assets_root": assets_root,
        "agent_file": agent_file,
        "env_file": env_file,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        formatted = ", ".join(f"{name}={required[name]}" for name in missing)
        raise XLeRobotManiSkillError(
            f"XLeRobot ManiSkill files are missing or moved: {formatted}"
        )

    _ensure_mani_skill_runtime_installed()

    robots_module = importlib.import_module("mani_skill.agents.robots")
    agent_module = _load_module(
        "multido_xlerobot._sim.xlerobot_agent",
        agent_file,
        force_reload=force_reload,
    )

    xlerobot_cls = getattr(agent_module, "Xlerobot", None)
    if xlerobot_cls is None:
        raise XLeRobotManiSkillError(f"`Xlerobot` class not found in {agent_file}")

    # Point the registered agent at the URDF in the fork so no site-packages patching
    # is required.
    xlerobot_cls.urdf_path = str(assets_root / "xlerobot.urdf")
    setattr(robots_module, "Xlerobot", xlerobot_cls)
    if hasattr(robots_module, "__all__") and "Xlerobot" not in robots_module.__all__:
        robots_module.__all__ = list(robots_module.__all__) + ["Xlerobot"]

    env_module = _load_module(
        "multido_xlerobot._sim.scene_manipulation_env",
        env_file,
        force_reload=force_reload,
    )

    return XLeRobotManiSkillBootstrapResult(
        repo_root=root,
        sim_root=sim_root,
        assets_root=assets_root,
        agent_module=agent_module,
        env_module=env_module,
    )


def run_keyboard_play_demo(
    *,
    repo_root: str | Path = DEFAULT_XLEROBOT_FORK_ROOT,
    demo: str = "ee_keyboard",
    env_id: str = "SceneManipulation-v1",
    robot_uid: str = "xlerobot",
    control_mode: str | None = None,
    obs_mode: str | None = None,
    render_mode: str = "human",
    shader: str = "default",
    sim_backend: str = "auto",
    num_envs: int = 1,
    record_dir: str | None = None,
    force_reload: bool = False,
) -> Any:
    bootstrap_xlerobot_maniskill(repo_root, force_reload=force_reload)
    demo_module = _load_demo_module(Path(repo_root).expanduser().resolve(), demo)

    args = demo_module.Args()
    args.env_id = env_id
    args.robot_uids = robot_uid
    args.render_mode = render_mode
    args.shader = shader
    args.sim_backend = sim_backend
    args.num_envs = num_envs

    if control_mode is not None:
        args.control_mode = control_mode
    if obs_mode is not None:
        args.obs_mode = obs_mode
    if record_dir is not None and hasattr(args, "record_dir"):
        args.record_dir = record_dir

    return demo_module.main(args)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the XLeRobot keyboard-control play mode in ManiSkill without "
            "manually copying files into site-packages."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_XLEROBOT_FORK_ROOT),
        help="Path to the local xlerobot_forked repository.",
    )
    parser.add_argument(
        "--demo",
        choices=("ee_keyboard", "joint_control", "camera_rerun"),
        default="ee_keyboard",
        help="Which local XLeRobot ManiSkill demo to run.",
    )
    parser.add_argument(
        "--env-id",
        default="SceneManipulation-v1",
        help=(
            "Gym environment id. `SceneManipulation-v1` is the safest default "
            "because it is registered locally at runtime."
        ),
    )
    parser.add_argument(
        "--robot-uid",
        default="xlerobot",
        help="Robot uid to pass to ManiSkill.",
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Override the demo's default control mode.",
    )
    parser.add_argument(
        "--obs-mode",
        default=None,
        help="Override the demo's default observation mode.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        help="ManiSkill render mode. Use `human` for interactive play.",
    )
    parser.add_argument(
        "--shader",
        default="default",
        help="Shader pack to use. `default` is a good starting point.",
    )
    parser.add_argument(
        "--sim-backend",
        default="auto",
        help="Simulation backend to pass through to ManiSkill.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to run.",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Optional ManiSkill recording directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the local setup and exit without launching the demo.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reloading local XLeRobot sim modules before launching.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        result = bootstrap_xlerobot_maniskill(
            args.repo_root,
            force_reload=args.force_reload,
        )
    except XLeRobotManiSkillError as exc:
        parser.exit(
            2,
            f"Setup error: {exc}\n"
            "Tip: run scripts/setup_xlerobot_maniskill_env.sh first, and make sure "
            "you launch this command from that environment.\n",
        )

    if args.check:
        print("XLeRobot ManiSkill bootstrap succeeded.")
        print(f"repo_root: {result.repo_root}")
        print(f"sim_root: {result.sim_root}")
        print(f"assets_root: {result.assets_root}")
        print("registered_agent: xlerobot")
        print("registered_env: SceneManipulation-v1")
        return 0

    run_keyboard_play_demo(
        repo_root=args.repo_root,
        demo=args.demo,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        shader=args.shader,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        record_dir=args.record_dir,
        force_reload=args.force_reload,
    )
    return 0


def _ensure_mani_skill_runtime_installed() -> None:
    missing: list[str] = []
    for module_name in ("mani_skill", "gymnasium", "sapien", "pygame", "tyro"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                missing.append(module_name)
            else:
                missing.append(f"{module_name} (import failed because {exc.name} is missing)")
        except Exception as exc:  # pragma: no cover - defensive import diagnostics
            missing.append(f"{module_name} (import failed: {exc})")

    if missing:
        raise XLeRobotManiSkillError(
            "Missing Python runtime dependencies for the play setup: "
            + ", ".join(missing)
        )


def _load_demo_module(repo_root: Path, demo: str) -> ModuleType:
    mapping = {
        "ee_keyboard": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action_ee_keyboard.py",
        "joint_control": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action.py",
        "camera_rerun": repo_root
        / "simulation"
        / "Maniskill"
        / "examples"
        / "demo_ctrl_action_ee_cam_rerun.py",
    }
    file_path = mapping[demo]
    return _load_module(
        f"multido_xlerobot._sim.demo_{demo}",
        file_path,
        force_reload=False,
    )


def _load_module(module_name: str, file_path: Path, *, force_reload: bool) -> ModuleType:
    if not file_path.exists():
        raise XLeRobotManiSkillError(f"Missing module file: {file_path}")

    if force_reload and module_name in sys.modules:
        del sys.modules[module_name]
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise XLeRobotManiSkillError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
