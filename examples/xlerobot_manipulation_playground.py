from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_playground.launcher import default_sim_python_bin, exec_python_module, resolve_repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified manipulation playground for XLeRobot. "
            "Choose sim or real, and keyboard or VR control."
        )
    )
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    parser.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sim-python-bin", default=None)
    parser.add_argument("--runtime-python-bin", default=sys.executable)

    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--speed-profile", choices=("normal", "fast"), default="normal")
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")

    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--xlevr-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)

    if args.backend == "sim":
        python_bin = args.sim_python_bin or default_sim_python_bin(REPO_ROOT)
        backend_args = [
            "manipulate",
            "--controller",
            args.controller,
            "--repo-root",
            str(repo_root),
            "--env-id",
            args.env_id,
            "--robot-uid",
            args.robot_uid,
            "--render-mode",
            args.render_mode,
            "--shader",
            args.shader,
            "--sim-backend",
            args.sim_backend,
            "--num-envs",
            str(args.num_envs),
        ]
        backend_args.extend(["--speed-profile", args.speed_profile])
        if args.control_mode is not None:
            backend_args.extend(["--control-mode", args.control_mode])
        if args.obs_mode is not None:
            backend_args.extend(["--obs-mode", args.obs_mode])
        if args.force_reload:
            backend_args.append("--force-reload")
        return exec_python_module(
            "xlerobot_playground.sim_backend",
            python_bin=python_bin,
            argv=backend_args,
            cwd=REPO_ROOT,
        )

    backend_args = [
        "manipulate",
        "--controller",
        args.controller,
        "--repo-root",
        str(repo_root),
        "--port1",
        args.port1,
        "--port2",
        args.port2,
        "--fps",
        str(args.fps),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--camera-fps",
        str(args.camera_fps),
    ]
    for camera in args.camera:
        backend_args.extend(["--camera", camera])
    if args.use_degrees:
        backend_args.append("--use-degrees")
    if args.xlevr_path is not None:
        backend_args.extend(["--xlevr-path", args.xlevr_path])

    return exec_python_module(
        "xlerobot_playground.real_backend",
        python_bin=args.runtime_python_bin,
        argv=backend_args,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
