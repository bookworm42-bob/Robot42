from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_playground.real_exploration_runtime import (
    RealXLeRobotDirectRuntime,
    RealXLeRobotRuntimeConfig,
)


@dataclass(frozen=True)
class RobotBrainAgentConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    repo_root: str = str(resolve_xlerobot_repo_root())
    robot_kind: str = "xlerobot_2wheels"
    port1: str = "/dev/tty.usbmodem5B140330101"
    port2: str = "/dev/tty.usbmodem5B140332271"
    use_degrees: bool = False
    allow_motion_commands: bool = False
    max_linear_m_s: float = 0.05
    max_angular_rad_s: float = 0.20
    debug_motion: bool = False
    calibration_prompt_response: str | None = ""
    orbbec_output_dir: Path = Path("artifacts/orbbec_rgbd")
    rgb_filename: str = "latest.ppm"
    depth_filename: str = "latest_depth.pgm"
    metadata_filename: str = "latest.json"
    imu_filename: str = "latest_imu.json"


class RobotBrainAgent:
    """Non-ROS hardware endpoint for the robot brain.

    The robot brain owns serial ports and local Orbbec capture files. The ROS
    offload computer can fetch RGB-D frames and forward velocity commands over
    this small HTTP API.
    """

    def __init__(self, config: RobotBrainAgentConfig, *, runtime: RealXLeRobotDirectRuntime | None = None) -> None:
        self.config = config
        runtime_config = RealXLeRobotRuntimeConfig(
            repo_root=config.repo_root,
            robot_kind=config.robot_kind,
            port1=config.port1,
            port2=config.port2,
            use_degrees=config.use_degrees,
            allow_motion_commands=config.allow_motion_commands,
            max_linear_m_s=config.max_linear_m_s,
            max_angular_rad_s=config.max_angular_rad_s,
            debug_motion=config.debug_motion,
            calibration_prompt_response=config.calibration_prompt_response,
        )
        self.runtime = runtime or RealXLeRobotDirectRuntime(runtime_config)
        self._motion_lock = threading.Lock()

    def velocity(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, Any]:
        if self.config.debug_motion:
            print(f"[robot_brain_agent] /cmd_vel requested linear={linear_m_s} angular={angular_rad_s}", flush=True)
        with self._motion_lock:
            if self.config.debug_motion:
                print("[robot_brain_agent] /cmd_vel acquired motion lock", flush=True)
            result = self.runtime.drive_velocity(linear_m_s=linear_m_s, angular_rad_s=angular_rad_s)
        if self.config.debug_motion:
            print(f"[robot_brain_agent] /cmd_vel done succeeded={result.succeeded}", flush=True)
        return {
            "succeeded": result.succeeded,
            "message": result.message,
            "metadata": result.metadata or {},
        }

    def stop(self) -> dict[str, Any]:
        if self.config.debug_motion:
            print("[robot_brain_agent] /stop requested", flush=True)
        with self._motion_lock:
            if self.config.debug_motion:
                print("[robot_brain_agent] /stop acquired motion lock", flush=True)
            result = self.runtime.stop()
        if self.config.debug_motion:
            print(f"[robot_brain_agent] /stop done succeeded={result.succeeded}", flush=True)
        return {
            "succeeded": result.succeeded,
            "message": result.message,
            "metadata": result.metadata or {},
        }

    def close(self) -> None:
        with self._motion_lock:
            try:
                self.runtime.stop()
            finally:
                self.runtime.close()

    def file_path(self, route: str) -> Path | None:
        if route == "/rgb":
            return self.config.orbbec_output_dir / self.config.rgb_filename
        if route == "/depth":
            return self.config.orbbec_output_dir / self.config.depth_filename
        if route == "/metadata":
            return self.config.orbbec_output_dir / self.config.metadata_filename
        if route == "/imu":
            return self.config.orbbec_output_dir / self.config.imu_filename
        return None


def make_handler(agent: RobotBrainAgent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotBrainAgent/0.1"

        def do_GET(self) -> None:
            self._serve_get_or_head(include_body=True)

        def do_HEAD(self) -> None:
            self._serve_get_or_head(include_body=False)

        def _serve_get_or_head(self, *, include_body: bool) -> None:
            if self.path == "/health":
                self._send_json(
                    {
                        "ok": True,
                        "robot_kind": agent.config.robot_kind,
                        "motion_enabled": agent.config.allow_motion_commands,
                    },
                    include_body=include_body,
                )
                return
            path = agent.file_path(self.path)
            if path is None:
                self.send_error(404, "Unknown route")
                return
            path = path.expanduser().resolve()
            if not path.exists():
                self.send_error(404, f"File not ready: {path.name}")
                return
            content_type = {
                "/rgb": "image/x-portable-pixmap",
                "/depth": "image/x-portable-graymap",
                "/metadata": "application/json",
                "/imu": "application/json",
            }.get(self.path, "application/octet-stream")
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if include_body:
                self.wfile.write(data)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/cmd_vel":
                    response = agent.velocity(
                        linear_m_s=float(payload.get("linear_m_s", 0.0)),
                        angular_rad_s=float(payload.get("angular_rad_s", 0.0)),
                    )
                    self._send_json(response)
                    return
                if self.path == "/stop":
                    self._send_json(agent.stop())
                    return
                self.send_error(404, "Unknown route")
            except Exception as exc:
                self.send_error(500, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], *, include_body: bool = True) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if include_body:
                self.wfile.write(data)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the non-ROS XLeRobot brain HTTP agent.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot_2wheels")
    parser.add_argument("--port1", default="/dev/tty.usbmodem5B140330101")
    parser.add_argument("--port2", default="/dev/tty.usbmodem5B140332271")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--allow-motion-commands", action="store_true")
    parser.add_argument("--max-linear-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.20)
    parser.add_argument("--debug-motion", action="store_true")
    parser.add_argument(
        "--calibration-prompt-response",
        default="",
        help=(
            "Automatic response for the XLeRobot calibration prompt during connect. "
            "Default empty response restores calibration from file. Use 'c' for manual calibration."
        ),
    )
    parser.add_argument(
        "--interactive-calibration",
        action="store_true",
        help="Do not auto-answer the XLeRobot calibration prompt.",
    )
    parser.add_argument("--orbbec-output-dir", default="artifacts/orbbec_rgbd")
    parser.add_argument("--rgb-filename", default="latest.ppm")
    parser.add_argument("--depth-filename", default="latest_depth.pgm")
    parser.add_argument("--metadata-filename", default="latest.json")
    parser.add_argument(
        "--imu-filename",
        default="latest_imu.json",
        help="JSON file that carries the latest Orbbec IMU sample. Default uses the dedicated high-rate latest_imu.json stream.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RobotBrainAgentConfig:
    return RobotBrainAgentConfig(
        host=args.host,
        port=args.port,
        repo_root=args.repo_root,
        robot_kind=args.robot_kind,
        port1=args.port1,
        port2=args.port2,
        use_degrees=args.use_degrees,
        allow_motion_commands=args.allow_motion_commands,
        max_linear_m_s=args.max_linear_m_s,
        max_angular_rad_s=args.max_angular_rad_s,
        debug_motion=args.debug_motion,
        calibration_prompt_response=None if args.interactive_calibration else args.calibration_prompt_response,
        orbbec_output_dir=Path(args.orbbec_output_dir),
        rgb_filename=args.rgb_filename,
        depth_filename=args.depth_filename,
        metadata_filename=args.metadata_filename,
        imu_filename=args.imu_filename,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    agent = RobotBrainAgent(config)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(agent))
    print(
        "Robot brain agent ready: "
        f"http://{config.host}:{config.port} robot={config.robot_kind} "
        f"motion_enabled={config.allow_motion_commands} orbbec={config.orbbec_output_dir}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
