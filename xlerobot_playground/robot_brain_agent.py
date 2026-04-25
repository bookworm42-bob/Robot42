from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_playground.imu_transport import parse_imu_json
from xlerobot_playground.real_exploration_runtime import (
    RealXLeRobotDirectRuntime,
    RealXLeRobotRuntimeConfig,
)
from xlerobot_playground.rgbd_transport import (
    PackedRgbdFrame,
    depth_payload_to_pgm,
    rgb_payload_to_ppm,
    unpack_rgbd_frame,
)

try:
    from aiohttp import WSMsgType, web
except Exception as exc:  # pragma: no cover - runtime dependency guard.
    WSMsgType = None
    web = None
    AIOHTTP_IMPORT_ERROR: Exception | None = exc
else:
    AIOHTTP_IMPORT_ERROR = None


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
    imu_udp_host: str = "127.0.0.1"
    imu_udp_port: int = 8766
    imu_ws_client_queue_size: int = 64
    imu_log_every: int = 200
    camera_max_frame_bytes: int = 16 * 1024 * 1024
    camera_log_every: int = 30


class ImuStreamState:
    def __init__(self, *, queue_size: int, log_every: int) -> None:
        self.queue_size = max(1, int(queue_size))
        self.log_every = max(1, int(log_every))
        self.latest_sample: dict[str, Any] | None = None
        self.latest_json: str | None = None
        self._clients: set[asyncio.Queue[str]] = set()
        self._received_count = 0
        self._sent_count = 0
        self._queue_drop_count = 0
        self._last_rx_log_at = time.monotonic()
        self._last_rx_log_count = 0
        self._last_tx_log_at = time.monotonic()
        self._last_tx_log_count = 0
        self._last_received_monotonic_s: float | None = None

    def publish(self, sample: dict[str, Any]) -> None:
        payload = json.dumps(sample, separators=(",", ":"))
        self.latest_sample = sample
        self.latest_json = payload
        self._received_count += 1
        self._last_received_monotonic_s = time.monotonic()
        for queue in tuple(self._clients):
            if queue.full():
                try:
                    queue.get_nowait()
                    self._queue_drop_count += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._queue_drop_count += 1
        if self._received_count % self.log_every == 0:
            now = time.monotonic()
            dt = max(now - self._last_rx_log_at, 1e-6)
            delta = self._received_count - self._last_rx_log_count
            rate_hz = delta / dt
            print(
                "[robot_brain_agent] IMU rx "
                f"rate~={rate_hz:.1f}Hz total={self._received_count} "
                f"clients={len(self._clients)} queue_drops={self._queue_drop_count}",
                flush=True,
            )
            self._last_rx_log_at = now
            self._last_rx_log_count = self._received_count

    def note_sent(self) -> None:
        self._sent_count += 1
        if self._sent_count % self.log_every == 0:
            now = time.monotonic()
            dt = max(now - self._last_tx_log_at, 1e-6)
            delta = self._sent_count - self._last_tx_log_count
            rate_hz = delta / dt
            print(
                "[robot_brain_agent] IMU ws send "
                f"rate~={rate_hz:.1f}Hz total={self._sent_count} "
                f"clients={len(self._clients)} queue_drops={self._queue_drop_count}",
                flush=True,
            )
            self._last_tx_log_at = now
            self._last_tx_log_count = self._sent_count

    def register_client(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_size)
        self._clients.add(queue)
        if self.latest_json is not None:
            queue.put_nowait(self.latest_json)
        return queue

    def unregister_client(self, queue: asyncio.Queue[str]) -> None:
        self._clients.discard(queue)

    def stats(self) -> dict[str, Any]:
        imu_age_s = None
        if self._last_received_monotonic_s is not None:
            imu_age_s = max(time.monotonic() - self._last_received_monotonic_s, 0.0)
        return {
            "ready": self.latest_sample is not None,
            "age_s": None if imu_age_s is None else round(imu_age_s, 3),
            "received_count": self._received_count,
            "sent_count": self._sent_count,
            "client_count": len(self._clients),
            "queue_drop_count": self._queue_drop_count,
            "latest_timestamp_s": None
            if self.latest_sample is None
            else self.latest_sample.get("timestamp_s"),
        }


class RgbdStreamState:
    def __init__(self, *, log_every: int) -> None:
        self.log_every = max(1, int(log_every))
        self.latest_frame: PackedRgbdFrame | None = None
        self.latest_payload: bytes | None = None
        self._received_count = 0
        self._last_rx_log_at = time.monotonic()
        self._last_rx_log_count = 0
        self._last_received_monotonic_s: float | None = None

    def publish_payload(self, payload: bytes) -> PackedRgbdFrame:
        frame = unpack_rgbd_frame(payload)
        self.latest_frame = frame
        self.latest_payload = payload
        self._received_count += 1
        self._last_received_monotonic_s = time.monotonic()
        if self._received_count % self.log_every == 0:
            now = time.monotonic()
            dt = max(now - self._last_rx_log_at, 1e-6)
            delta = self._received_count - self._last_rx_log_count
            print(
                "[robot_brain_agent] RGB-D rx "
                f"rate~={delta / dt:.1f}Hz total={self._received_count} "
                f"rgb={frame.rgb_width}x{frame.rgb_height} "
                f"depth={frame.depth_width or 0}x{frame.depth_height or 0}",
                flush=True,
            )
            self._last_rx_log_at = now
            self._last_rx_log_count = self._received_count
        return frame

    def rgb_ppm(self) -> bytes | None:
        frame = self.latest_frame
        if frame is None:
            return None
        return rgb_payload_to_ppm(frame.rgb, width=frame.rgb_width, height=frame.rgb_height)

    def depth_pgm(self) -> bytes | None:
        frame = self.latest_frame
        if frame is None or frame.depth_be is None or frame.depth_width is None or frame.depth_height is None:
            return None
        return depth_payload_to_pgm(frame.depth_be, width=frame.depth_width, height=frame.depth_height)

    def metadata_json(self) -> bytes | None:
        frame = self.latest_frame
        if frame is None:
            return None
        return json.dumps(self.stats(), sort_keys=True).encode("utf-8")

    def stats(self) -> dict[str, Any]:
        age_s = None
        if self._last_received_monotonic_s is not None:
            age_s = max(time.monotonic() - self._last_received_monotonic_s, 0.0)
        frame = self.latest_frame
        return {
            "ready": frame is not None,
            "age_s": None if age_s is None else round(age_s, 3),
            "received_count": self._received_count,
            "frame_index": None if frame is None else frame.frame_index,
            "timestamp_s": None if frame is None else frame.timestamp_s,
            "rgb": None if frame is None else {"width": frame.rgb_width, "height": frame.rgb_height},
            "depth": None
            if frame is None or frame.depth_width is None or frame.depth_height is None
            else {"width": frame.depth_width, "height": frame.depth_height},
        }


class RobotBrainAgent:
    """Non-ROS hardware endpoint for the robot brain."""

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
        self.imu_stream = ImuStreamState(
            queue_size=config.imu_ws_client_queue_size,
            log_every=config.imu_log_every,
        )
        self.rgbd_stream = RgbdStreamState(log_every=config.camera_log_every)

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

    def ingest_imu_datagram(self, data: bytes) -> None:
        sample = parse_imu_json(data)
        if sample is None:
            return
        self.imu_stream.publish(sample)

    def ingest_rgbd_payload(self, data: bytes) -> PackedRgbdFrame:
        return self.rgbd_stream.publish_payload(data)

    def imu_snapshot(self) -> dict[str, Any] | None:
        return self.imu_stream.latest_sample

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
        return None


class _ImuUdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, agent: RobotBrainAgent) -> None:
        self.agent = agent

    def datagram_received(self, data: bytes, addr: tuple[str, int] | tuple[str, int, int, int]) -> None:
        try:
            self.agent.ingest_imu_datagram(data)
        except Exception as exc:
            print(f"[robot_brain_agent] Failed to decode IMU datagram from {addr}: {exc}", flush=True)

    def error_received(self, exc: Exception) -> None:
        print(f"[robot_brain_agent] IMU UDP socket error: {exc}", flush=True)


def require_aiohttp() -> None:
    if AIOHTTP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "robot_brain_agent requires `aiohttp` for the shared HTTP + websocket server. "
            "Install it with `python -m pip install aiohttp`."
        ) from AIOHTTP_IMPORT_ERROR


async def _read_route_bytes(agent: RobotBrainAgent, route: str) -> bytes:
    if route == "/rgbd":
        if agent.rgbd_stream.latest_payload is not None:
            return agent.rgbd_stream.latest_payload
    elif route == "/rgb":
        rgb = agent.rgbd_stream.rgb_ppm()
        if rgb is not None:
            return rgb
    elif route == "/depth":
        depth = agent.rgbd_stream.depth_pgm()
        if depth is not None:
            return depth
    elif route == "/metadata":
        metadata = agent.rgbd_stream.metadata_json()
        if metadata is not None:
            return metadata
    path = agent.file_path(route)
    if path is None:
        raise web.HTTPNotFound(text="Unknown route")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise web.HTTPNotFound(text=f"File not ready: {resolved.name}")
    return await asyncio.to_thread(resolved.read_bytes)


async def _handle_health(_request: web.Request) -> web.Response:
    agent: RobotBrainAgent = _request.app["agent"]
    return web.json_response(
        {
            "ok": True,
            "robot_kind": agent.config.robot_kind,
            "motion_enabled": agent.config.allow_motion_commands,
            "imu_udp_port": agent.config.imu_udp_port,
            "imu_ws_path": "/ws/imu",
            "imu": agent.imu_stream.stats(),
            "rgbd": agent.rgbd_stream.stats(),
        }
    )


async def _handle_static_file(request: web.Request) -> web.Response:
    agent: RobotBrainAgent = request.app["agent"]
    data = await _read_route_bytes(agent, request.path)
    content_type = {
        "/rgb": "image/x-portable-pixmap",
        "/depth": "image/x-portable-graymap",
        "/rgbd": "application/octet-stream",
        "/metadata": "application/json",
    }.get(request.path, "application/octet-stream")
    return web.Response(body=data, content_type=content_type)


async def _handle_rgbd_ingest(request: web.Request) -> web.Response:
    agent: RobotBrainAgent = request.app["agent"]
    data = await request.read()
    frame = agent.ingest_rgbd_payload(data)
    return web.json_response(
        {
            "ok": True,
            "frame_index": frame.frame_index,
            "timestamp_s": frame.timestamp_s,
            "rgb": {"width": frame.rgb_width, "height": frame.rgb_height},
            "depth": None
            if frame.depth_width is None or frame.depth_height is None
            else {"width": frame.depth_width, "height": frame.depth_height},
        }
    )


async def _handle_imu_snapshot(request: web.Request) -> web.Response:
    agent: RobotBrainAgent = request.app["agent"]
    if agent.imu_stream.latest_json is None:
        raise web.HTTPNotFound(text="IMU sample not ready")
    return web.Response(body=agent.imu_stream.latest_json.encode("utf-8"), content_type="application/json")


async def _handle_cmd_vel(request: web.Request) -> web.Response:
    agent: RobotBrainAgent = request.app["agent"]
    raw = await request.read() if request.can_read_body else b""
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    response = await asyncio.to_thread(
        agent.velocity,
        linear_m_s=float(payload.get("linear_m_s", 0.0)),
        angular_rad_s=float(payload.get("angular_rad_s", 0.0)),
    )
    return web.json_response(response)


async def _handle_stop(request: web.Request) -> web.Response:
    agent: RobotBrainAgent = request.app["agent"]
    if request.can_read_body:
        try:
            await request.read()
        except Exception:
            pass
    response = await asyncio.to_thread(agent.stop)
    return web.json_response(response)


async def _imu_sender(agent: RobotBrainAgent, ws: web.WebSocketResponse, queue: asyncio.Queue[str]) -> None:
    while True:
        payload = await queue.get()
        await ws.send_str(payload)
        agent.imu_stream.note_sent()


async def _handle_imu_websocket(request: web.Request) -> web.WebSocketResponse:
    agent: RobotBrainAgent = request.app["agent"]
    ws = web.WebSocketResponse(heartbeat=20.0)
    await ws.prepare(request)
    queue = agent.imu_stream.register_client()
    print(f"[robot_brain_agent] IMU websocket connected clients={len(agent.imu_stream._clients)}", flush=True)
    sender = asyncio.create_task(_imu_sender(agent, ws, queue))
    try:
        async for message in ws:
            if message.type == WSMsgType.ERROR:
                raise ws.exception() or RuntimeError("websocket error")
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                break
    finally:
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
        agent.imu_stream.unregister_client(queue)
        print(f"[robot_brain_agent] IMU websocket disconnected clients={len(agent.imu_stream._clients)}", flush=True)
    return ws


async def _runtime_context(app: web.Application) -> Any:
    agent: RobotBrainAgent = app["agent"]
    loop = asyncio.get_running_loop()
    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: _ImuUdpProtocol(agent),
        local_addr=(agent.config.imu_udp_host, agent.config.imu_udp_port),
    )
    app["imu_udp_transport"] = transport
    print(
        "[robot_brain_agent] IMU UDP listener ready: "
        f"{agent.config.imu_udp_host}:{agent.config.imu_udp_port}",
        flush=True,
    )
    try:
        yield
    finally:
        transport.close()
        await asyncio.to_thread(agent.close)


def build_app(agent: RobotBrainAgent) -> web.Application:
    require_aiohttp()
    app = web.Application(client_max_size=agent.config.camera_max_frame_bytes)
    app["agent"] = agent
    app.cleanup_ctx.append(_runtime_context)
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/rgbd", _handle_static_file)
    app.router.add_get("/rgb", _handle_static_file)
    app.router.add_get("/depth", _handle_static_file)
    app.router.add_get("/metadata", _handle_static_file)
    app.router.add_post("/camera/rgbd", _handle_rgbd_ingest)
    app.router.add_get("/imu", _handle_imu_snapshot)
    app.router.add_get("/ws/imu", _handle_imu_websocket)
    app.router.add_post("/cmd_vel", _handle_cmd_vel)
    app.router.add_post("/stop", _handle_stop)
    return app


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
    parser.add_argument("--imu-udp-host", default="127.0.0.1")
    parser.add_argument("--imu-udp-port", type=int, default=8766)
    parser.add_argument("--imu-ws-client-queue-size", type=int, default=64)
    parser.add_argument("--imu-log-every", type=int, default=200)
    parser.add_argument("--camera-max-frame-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--camera-log-every", type=int, default=30)
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
        imu_udp_host=args.imu_udp_host,
        imu_udp_port=args.imu_udp_port,
        imu_ws_client_queue_size=args.imu_ws_client_queue_size,
        imu_log_every=args.imu_log_every,
        camera_max_frame_bytes=args.camera_max_frame_bytes,
        camera_log_every=args.camera_log_every,
    )


def main(argv: list[str] | None = None) -> int:
    require_aiohttp()
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    agent = RobotBrainAgent(config)
    print(
        "Robot brain agent ready: "
        f"http://{config.host}:{config.port} robot={config.robot_kind} "
        f"motion_enabled={config.allow_motion_commands} orbbec={config.orbbec_output_dir} "
        f"imu_udp={config.imu_udp_host}:{config.imu_udp_port} imu_ws=/ws/imu",
        flush=True,
    )
    web.run_app(build_app(agent), host=config.host, port=config.port, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
