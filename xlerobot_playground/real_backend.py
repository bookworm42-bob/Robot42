from __future__ import annotations

import argparse
import importlib.util
import ssl
import threading
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import bootstrap_xlerobot, resolve_xlerobot_repo_root


@dataclass(frozen=True)
class CameraSpec:
    name: str
    driver: str
    source: str


@dataclass
class RecordingSession:
    dataset: Any
    task: str
    active: bool = False


@dataclass(frozen=True)
class OrbbecRgbConfig:
    enabled: bool
    launch_capture: bool
    capture_bin: Path
    output_dir: Path
    width: int
    height: int
    fps: int
    timeout_ms: int


_VR_CAMERA_FRAMES: dict[str, tuple[bytes, int, int, float]] = {}
_VR_CAMERA_FRAMES_LOCK = threading.Lock()
_VR_CAMERA_JPEGS: dict[str, tuple[bytes, int, int, float]] = {}
_VR_CAMERA_JPEGS_LOCK = threading.Lock()
_ORBBEC_JPEG_CACHE: tuple[Path, int, int, bytes, float] | None = None
_ORBBEC_JPEG_CACHE_LOCK = threading.Lock()
_JPEG_QUALITY = 85
_MJPEG_BOUNDARY = b"frame"


@dataclass(frozen=True)
class VRRecordingControls:
    toggle_recording: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


@dataclass(frozen=True)
class VRRecordingDecision:
    start_recording: bool = False
    save_episode: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend launcher for XLeRobot real teleop and local LeRobot recording."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manipulate = subparsers.add_parser("manipulate", help="Launch real teleoperation.")
    _add_shared_args(manipulate)
    manipulate.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")

    record = subparsers.add_parser("record", help="Launch real teleop with local LeRobot recording.")
    _add_shared_args(record)
    record.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    record.add_argument("--dataset-id", default="local/xlerobot_playground")
    record.add_argument("--dataset-root", default="./datasets")
    record.add_argument("--task", default="XLeRobot teleoperation")
    record.add_argument("--use-videos", action="store_true")
    record.add_argument("--start-key", default="[")
    record.add_argument("--stop-key", default="]")
    record.add_argument("--quit-key", default="\\")
    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(resolve_xlerobot_repo_root()))
    parser.add_argument("--robot-kind", choices=("xlerobot", "xlerobot_2wheels"), default="xlerobot")
    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=DRIVER:SOURCE",
        help=(
            "Camera config. Example: `head=realsense:125322060037` or "
            "`left_wrist=opencv:/dev/video0`."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--xlevr-path", default=None)
    parser.add_argument(
        "--orbbec-rgb-vr",
        action="store_true",
        help="Show the Orbbec Gemini 2 RGB stream in the Quest/XLeVR scene.",
    )
    parser.add_argument(
        "--orbbec-capture-bin",
        default=None,
        help="Path to the native orbbec_rgb_test binary.",
    )
    parser.add_argument(
        "--orbbec-no-launch",
        action="store_true",
        help="Serve the Quest overlay from an already-running Orbbec RGB sidecar.",
    )
    parser.add_argument("--orbbec-output-dir", default="artifacts/orbbec_rgb")
    parser.add_argument("--orbbec-width", type=int, default=640)
    parser.add_argument("--orbbec-height", type=int, default=480)
    parser.add_argument("--orbbec-fps", type=int, default=30)
    parser.add_argument("--orbbec-timeout-ms", type=int, default=1000)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_xlerobot(args.repo_root)

    interface = XLeRobotInterface(args.repo_root)
    if args.robot_kind == "xlerobot_2wheels":
        config_cls, robot_cls = interface.robot_2wheels_classes()
    else:
        config_cls, robot_cls = interface.robot_classes()
    robot_config = config_cls(
        port1=args.port1,
        port2=args.port2,
        cameras=_build_camera_configs(
            args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        use_degrees=args.use_degrees,
    )
    robot = robot_cls(robot_config)

    recording = None
    if args.mode == "record":
        recording = RecordingSession(
            dataset=_create_dataset(
                robot,
                dataset_id=args.dataset_id,
                dataset_root=args.dataset_root,
                fps=args.fps,
                use_videos=args.use_videos,
            ),
            task=args.task,
        )

    if args.controller == "keyboard":
        return _run_keyboard_backend(
            repo_root=Path(args.repo_root).expanduser().resolve(),
            robot=robot,
            fps=args.fps,
            recording=recording,
            start_key=getattr(args, "start_key", "["),
            stop_key=getattr(args, "stop_key", "]"),
            quit_key=getattr(args, "quit_key", "\\"),
        )
    return _run_vr_backend(
        interface=interface,
        robot=robot,
        fps=args.fps,
        recording=recording,
        start_key=getattr(args, "start_key", "["),
        stop_key=getattr(args, "stop_key", "]"),
        quit_key=getattr(args, "quit_key", "\\"),
        xlevr_path=args.xlevr_path,
        orbbec_rgb=OrbbecRgbConfig(
            enabled=args.orbbec_rgb_vr,
            launch_capture=not args.orbbec_no_launch,
            capture_bin=Path(
                args.orbbec_capture_bin
                or Path(__file__).resolve().parents[1] / "build" / "orbbec_rgb_test" / "orbbec_rgb_test"
            ).expanduser().resolve(),
            output_dir=Path(args.orbbec_output_dir).expanduser().resolve(),
            width=args.orbbec_width,
            height=args.orbbec_height,
            fps=args.orbbec_fps,
            timeout_ms=args.orbbec_timeout_ms,
        ),
    )


def _build_camera_configs(
    camera_specs: list[str],
    *,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras: dict[str, Any] = {}
    for raw_spec in camera_specs:
        spec = _parse_camera_spec(raw_spec)
        if spec.driver == "opencv":
            source: Any = int(spec.source) if spec.source.isdigit() else spec.source
            cameras[spec.name] = OpenCVCameraConfig(
                index_or_path=source,
                fps=fps,
                width=width,
                height=height,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            continue
        if spec.driver == "realsense":
            cameras[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=spec.source,
                fps=fps,
                width=width,
                height=height,
                color_mode=ColorMode.BGR,
                rotation=Cv2Rotation.NO_ROTATION,
                use_depth=True,
            )
            continue
        raise ValueError(f"Unsupported camera driver `{spec.driver}` in `{raw_spec}`")
    return cameras


def _parse_camera_spec(raw_spec: str) -> CameraSpec:
    if "=" not in raw_spec or ":" not in raw_spec:
        raise ValueError(
            f"Invalid camera spec `{raw_spec}`. Use `NAME=DRIVER:SOURCE`, "
            "for example `head=realsense:125322060037`."
        )
    name, remainder = raw_spec.split("=", 1)
    driver, source = remainder.split(":", 1)
    return CameraSpec(name=name.strip(), driver=driver.strip(), source=source.strip())


def _start_orbbec_rgb_sidecar(config: OrbbecRgbConfig) -> subprocess.Popen[bytes] | None:
    if not config.enabled or not config.launch_capture:
        return None
    capture_bin = config.capture_bin.resolve()
    if not capture_bin.exists():
        raise FileNotFoundError(
            f"Orbbec RGB capture binary not found: {capture_bin}. "
            "Build it with `cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test "
            "-DORBBEC_SDK_ROOT=/Users/alin/orbbec/sdk && cmake --build build/orbbec_rgb_test`."
        )

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(capture_bin),
        "--frames",
        "0",
        "--latest-only",
        "--output-dir",
        str(output_dir),
        "--width",
        str(config.width),
        "--height",
        str(config.height),
        "--fps",
        str(config.fps),
        "--timeout-ms",
        str(config.timeout_ms),
        "--log-every",
        str(max(1, config.fps)),
    ]
    print(f"Starting Orbbec RGB sidecar: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _stop_orbbec_rgb_sidecar(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _enable_threaded_vr_http(vr_teleop: Any) -> bool:
    module_prefix = vr_teleop.__class__.__module__.rsplit(".", 1)[0]
    monitor_module = sys.modules.get(f"{module_prefix}.vr_monitor")
    if monitor_module is None or not hasattr(monitor_module, "http"):
        return False
    http_server_module = monitor_module.http.server
    if getattr(http_server_module, "HTTPServer", None) is http_server_module.ThreadingHTTPServer:
        return True
    http_server_module.ThreadingHTTPServer.daemon_threads = True
    http_server_module.HTTPServer = http_server_module.ThreadingHTTPServer
    return True


def _install_orbbec_vr_overlay(vr_teleop: Any, output_dir: Path, *, include_orbbec: bool) -> bool:
    monitor = getattr(vr_teleop, "vr_monitor", None)
    if monitor is None:
        return False
    handler_cls = getattr(sys.modules.get(monitor.__class__.__module__), "SimpleAPIHandler", None)
    if handler_cls is None:
        return False
    if getattr(handler_cls, "_orbbec_overlay_installed", False):
        handler_cls.orbbec_output_dir = output_dir.resolve()
        handler_cls.orbbec_overlay_include_orbbec = include_orbbec
        return True

    original_do_get = handler_cls.do_GET
    original_serve_file = handler_cls.serve_file
    handler_cls.orbbec_output_dir = output_dir.resolve()
    handler_cls.orbbec_overlay_include_orbbec = include_orbbec

    def do_GET(self):
        if self.path.startswith("/orbbec/latest.mjpg"):
            return _serve_orbbec_mjpeg_stream(self)
        if self.path.startswith("/orbbec/latest.ppm"):
            return _serve_orbbec_file(self, "latest.ppm", "image/x-portable-pixmap")
        if self.path.startswith("/orbbec/latest.json"):
            return _serve_orbbec_file(self, "latest.json", "application/json")
        if self.path.startswith("/vr-camera/"):
            return _serve_vr_camera_file(self)
        return original_do_get(self)

    def serve_file(self, filename, content_type):
        if filename == "web-ui/vr_app.js":
            try:
                web_root = getattr(self.server, "web_root_path", None)
                root = Path(web_root) if web_root else Path.cwd()
                js_path = root / filename
                content = js_path.read_text()
                content += "\n\n" + _orbbec_vr_overlay_js(
                    include_orbbec=getattr(handler_cls, "orbbec_overlay_include_orbbec", True)
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
                return
            except Exception as exc:
                print(f"Error injecting Orbbec VR overlay: {exc}")
        return original_serve_file(self, filename, content_type)

    handler_cls.do_GET = do_GET
    handler_cls.serve_file = serve_file
    handler_cls._orbbec_overlay_installed = True
    return True


def _publish_vr_camera_frames(obs: dict[str, Any]) -> None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return

    frames: dict[str, tuple[bytes, int, int, float]] = {}
    jpegs: dict[str, tuple[bytes, int, int, float]] = {}
    for name, value in obs.items():
        if name not in {"left_wrist", "right_wrist"}:
            continue
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] < 3:
            continue
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(array[:, :, :3])
        height, width = rgb.shape[:2]
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        captured_at = time.time()
        frames[name] = (header + rgb.tobytes(), width, height, captured_at)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if ok:
            jpegs[name] = (encoded.tobytes(), width, height, captured_at)

    if frames:
        with _VR_CAMERA_FRAMES_LOCK:
            _VR_CAMERA_FRAMES.update(frames)
    if jpegs:
        with _VR_CAMERA_JPEGS_LOCK:
            _VR_CAMERA_JPEGS.update(jpegs)


def _get_robot_observation(robot: Any, *, use_camera: bool = True) -> dict[str, Any]:
    try:
        return robot.get_observation(use_camera=use_camera)
    except TypeError:
        return robot.get_observation()


def _write_response(handler: Any, content: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(content)


def _serve_vr_camera_file(handler: Any) -> None:
    parsed = urllib.parse.urlparse(handler.path)
    rel = parsed.path.removeprefix("/vr-camera/")
    name, suffix = Path(rel).stem, Path(rel).suffix
    if not name or suffix not in {".ppm", ".json", ".mjpg"}:
        handler.send_error(404, "Unknown VR camera endpoint")
        return

    if suffix == ".mjpg":
        return _serve_vr_camera_mjpeg_stream(handler, name)

    with _VR_CAMERA_FRAMES_LOCK:
        frame = _VR_CAMERA_FRAMES.get(name)
    if frame is None:
        handler.send_error(404, f"VR camera frame not ready: {name}")
        return

    ppm, width, height, captured_at = frame
    if suffix == ".json":
        payload = (
            f'{{"name":"{name}","width":{width},"height":{height},'
            f'"captured_at":{captured_at:.6f}}}'
        ).encode("utf-8")
        content_type = "application/json"
        content = payload
    else:
        content_type = "image/x-portable-pixmap"
        content = ppm

    try:
        _write_response(handler, content, content_type)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return
    except Exception as exc:
        print(f"Error serving VR camera frame {name}: {exc}")
        handler.send_error(500, "Could not serve VR camera frame")


def _serve_vr_camera_mjpeg_stream(handler: Any, name: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.end_headers()
    _stream_jpeg_frames(handler, lambda: _latest_vr_camera_jpeg(name), fps=30)


def _latest_vr_camera_jpeg(name: str) -> tuple[bytes, int, int, float] | None:
    with _VR_CAMERA_JPEGS_LOCK:
        return _VR_CAMERA_JPEGS.get(name)


def _stream_jpeg_frames(
    handler: Any,
    latest_frame: Any,
    *,
    fps: int,
) -> None:
    last_timestamp = 0.0
    delay = 1.0 / max(1, fps)
    try:
        while True:
            frame = latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            jpeg, _width, _height, captured_at = frame
            if captured_at <= last_timestamp:
                time.sleep(min(0.01, delay))
                continue
            last_timestamp = captured_at
            handler.wfile.write(b"--" + _MJPEG_BOUNDARY + b"\r\n")
            handler.wfile.write(b"Content-Type: image/jpeg\r\n")
            handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
            handler.wfile.write(jpeg)
            handler.wfile.write(b"\r\n")
            handler.wfile.flush()
            time.sleep(delay)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError, ConnectionAbortedError):
        return


def _serve_orbbec_file(handler: Any, filename: str, content_type: str) -> None:
    output_dir = getattr(handler.__class__, "orbbec_output_dir", None)
    if output_dir is None:
        handler.send_error(404, "Orbbec RGB output directory is not configured")
        return
    path = Path(output_dir) / filename
    if not path.exists():
        handler.send_error(404, f"Orbbec RGB frame not ready: {filename}")
        return
    try:
        content = path.read_bytes()
        _write_response(handler, content, content_type)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
        return
    except Exception as exc:
        print(f"Error serving Orbbec RGB file {path}: {exc}")
        handler.send_error(500, "Could not serve Orbbec RGB frame")


def _serve_orbbec_mjpeg_stream(handler: Any) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.end_headers()
    _stream_jpeg_frames(handler, lambda: _latest_orbbec_jpeg(handler), fps=30)


def _latest_orbbec_jpeg(handler: Any) -> tuple[bytes, int, int, float] | None:
    output_dir = getattr(handler.__class__, "orbbec_output_dir", None)
    if output_dir is None:
        return None
    path = Path(output_dir) / "latest.ppm"
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    global _ORBBEC_JPEG_CACHE
    with _ORBBEC_JPEG_CACHE_LOCK:
        if (
            _ORBBEC_JPEG_CACHE is not None
            and _ORBBEC_JPEG_CACHE[0] == path
            and _ORBBEC_JPEG_CACHE[1] == stat.st_mtime_ns
            and _ORBBEC_JPEG_CACHE[2] == stat.st_size
        ):
            return (_ORBBEC_JPEG_CACHE[3], 0, 0, _ORBBEC_JPEG_CACHE[4])

    try:
        jpeg, width, height = _ppm_file_to_jpeg(path)
    except Exception:
        return None

    with _ORBBEC_JPEG_CACHE_LOCK:
        _ORBBEC_JPEG_CACHE = (path, stat.st_mtime_ns, stat.st_size, jpeg, stat.st_mtime)
    return (jpeg, width, height, stat.st_mtime)


def _ppm_file_to_jpeg(path: Path) -> tuple[bytes, int, int]:
    import cv2
    import numpy as np

    data = path.read_bytes()
    offset = 0

    def skip_ws_and_comments() -> None:
        nonlocal offset
        while offset < len(data):
            value = data[offset]
            if value == 35:
                while offset < len(data) and data[offset] != 10:
                    offset += 1
            elif value in (9, 10, 13, 32):
                offset += 1
            else:
                break

    def token() -> bytes:
        nonlocal offset
        skip_ws_and_comments()
        start = offset
        while offset < len(data) and data[offset] not in (9, 10, 13, 32, 35):
            offset += 1
        return data[start:offset]

    magic = token()
    width = int(token())
    height = int(token())
    max_value = int(token())
    skip_ws_and_comments()
    if magic != b"P6" or max_value != 255:
        raise ValueError(f"Unsupported PPM header in {path}")
    rgb = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=offset)
    rgb = rgb.reshape((height, width, 3))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Could not encode JPEG for {path}")
    return encoded.tobytes(), width, height


def _orbbec_vr_overlay_js(*, include_orbbec: bool) -> str:
    return r"""
(function () {
  const FEEDS = [
    {
      name: 'orbbec',
      url: '/orbbec/latest.mjpg',
      canvasId: 'orbbec-rgb-canvas',
      width: 0.8,
      height: 0.6,
      position: [0, -0.18, -1.05],
      markerPosition: [-0.42, 0.32, -1.04],
      logName: 'Orbbec RGB'
    },
    {
      name: 'left_wrist',
      url: '/vr-camera/left_wrist.mjpg',
      canvasId: 'left-wrist-rgb-canvas',
      width: 0.32,
      height: 0.24,
      position: [-0.17, -0.56, -1.02],
      markerPosition: [-0.34, -0.42, -1.01],
      logName: 'Left wrist'
    },
    {
      name: 'right_wrist',
      url: '/vr-camera/right_wrist.mjpg',
      canvasId: 'right-wrist-rgb-canvas',
      width: 0.32,
      height: 0.24,
      position: [0.17, -0.56, -1.02],
      markerPosition: [0.0, -0.42, -1.01],
      logName: 'Right wrist'
    }
  ];
  const INCLUDE_ORBBEC = __INCLUDE_ORBBEC__;
  const ACTIVE_FEEDS = FEEDS.filter(feed => INCLUDE_ORBBEC || feed.name !== 'orbbec');
  const overlays = new Map();
  let lastStatusLog = 0;

  function ensurePanel(feed) {
    const headset = document.querySelector('#headset');
    if (!headset || !headset.object3D || typeof THREE === 'undefined') return null;

    let canvas = document.getElementById(feed.canvasId);
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = feed.canvasId;
      canvas.width = 640;
      canvas.height = 480;
      canvas.style.display = 'none';
      document.body.appendChild(canvas);
    }

    if (!overlays.has(feed.name)) {
      const image = new Image();
      image.crossOrigin = 'anonymous';
      image.src = `${feed.url}?t=${Date.now()}`;

      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      if ('SRGBColorSpace' in THREE) texture.colorSpace = THREE.SRGBColorSpace;

      const material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        toneMapped: false
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(feed.width, feed.height), material);
      mesh.name = `${feed.name}-rgb-three-overlay`;
      mesh.position.set(feed.position[0], feed.position[1], feed.position[2]);
      mesh.renderOrder = 999;
      headset.object3D.add(mesh);

      const marker = new THREE.Mesh(
        new THREE.PlaneGeometry(feed.width * 0.1, feed.width * 0.1),
        new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide })
      );
      marker.name = `${feed.name}-rgb-marker`;
      marker.position.set(feed.markerPosition[0], feed.markerPosition[1], feed.markerPosition[2]);
      marker.renderOrder = 1000;
      headset.object3D.add(marker);

      overlays.set(feed.name, { canvas, texture, mesh, marker, image });
      console.log(`[${feed.logName}] MJPEG headset overlay created`);
    }

    return overlays.get(feed.name);
  }

  function updateFrame(feed) {
    const nodes = ensurePanel(feed);
    if (!nodes) return;
    try {
      const { canvas, texture, marker, image } = nodes;
      if (!image.naturalWidth || !image.naturalHeight) {
        throw new Error('MJPEG image not ready');
      }
      if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
      }
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      texture.image = canvas;
      texture.needsUpdate = true;
      if (marker) marker.material.color.setHex(0x00ff00);
      if (Date.now() - lastStatusLog > 3000) {
        console.log(`[${feed.logName}] Streaming ${canvas.width}x${canvas.height}`);
        lastStatusLog = Date.now();
      }
    } catch (error) {
      if (nodes.marker) nodes.marker.material.color.setHex(0xff0000);
      if (Date.now() - lastStatusLog > 3000) {
        console.warn(`[${feed.logName}] Waiting for frame`, error);
        lastStatusLog = Date.now();
      }
    }
  }

  function start() {
    function renderStreams() {
      ACTIVE_FEEDS.forEach(updateFrame);
      requestAnimationFrame(renderStreams);
    }
    renderStreams();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
""".replace("__INCLUDE_ORBBEC__", "true" if include_orbbec else "false")


def _run_keyboard_backend(
    *,
    repo_root: Path,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    import numpy as np

    keyboard_module = _load_example_module(
        repo_root / "software" / "examples" / "4_xlerobot_teleop_keyboard.py",
        "xlerobot_playground._real_keyboard_example",
    )

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    previous_pressed_keys: set[str] = set()

    robot.connect()
    init_rerun(session_name="xlerobot_real_keyboard_playground")
    keyboard.connect()

    obs = robot.get_observation()
    kin_left = keyboard_module.SO101Kinematics()
    kin_right = keyboard_module.SO101Kinematics()
    left_arm = keyboard_module.SimpleTeleopArm(kin_left, keyboard_module.LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = keyboard_module.SimpleTeleopArm(kin_right, keyboard_module.RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = keyboard_module.SimpleHeadControl(obs)

    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)
    _print_recording_guide(recording, start_key=start_key, stop_key=stop_key, quit_key=quit_key)

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(keyboard.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            left_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.LEFT_KEYMAP.items()
            }
            right_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.RIGHT_KEYMAP.items()
            }

            if left_key_state.get("triangle"):
                left_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if right_key_state.get("triangle"):
                right_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if left_key_state.get("reset"):
                left_arm.move_to_zero_position(robot)
                continue
            if right_key_state.get("reset"):
                right_arm.move_to_zero_position(robot)
                continue
            if "?" in pressed_keys:
                head_control.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            sent_action = robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        try:
            robot.disconnect()
        finally:
            if keyboard.is_connected:
                keyboard.disconnect()
    return 0


def _run_vr_backend(
    *,
    interface: XLeRobotInterface,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
    xlevr_path: str | None,
    orbbec_rgb: OrbbecRgbConfig,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    hotkeys = KeyboardTeleop(KeyboardTeleopConfig())
    previous_pressed_keys: set[str] = set()

    orbbec_process = _start_orbbec_rgb_sidecar(orbbec_rgb)
    robot.connect()
    init_rerun(session_name="xlerobot_real_vr_playground")
    hotkeys.connect()

    vr_overrides = {}
    if xlevr_path is not None:
        vr_overrides["xlevr_path"] = xlevr_path
    vr_teleop = interface.make_vr_teleop(**vr_overrides)
    if _enable_threaded_vr_http(vr_teleop):
        print("XLeVR HTTPS server patched to ThreadingHTTPServer for camera streams.")
    vr_teleop.connect(robot=robot)
    installed = _install_orbbec_vr_overlay(
        vr_teleop,
        orbbec_rgb.output_dir,
        include_orbbec=orbbec_rgb.enabled,
    )
    if installed:
        if orbbec_rgb.enabled:
            print("Orbbec RGB VR overlay enabled. Reload the Quest page if it was already open.")
        print("VR arm camera panels enabled for camera names: left_wrist and right_wrist.")
    elif orbbec_rgb.enabled:
        print("Orbbec RGB sidecar is running, but the XLeVR web overlay could not be installed.")
    vr_teleop.send_feedback()
    robot.send_action(vr_teleop.move_to_zero_position(robot))
    _print_recording_guide(
        recording,
        start_key=start_key,
        stop_key=stop_key,
        quit_key=quit_key,
        controller="vr",
    )

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(hotkeys.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            vr_decision = VRRecordingDecision()
            if recording is not None:
                vr_decision = _decide_vr_recording_action(
                    recording.active,
                    _map_vr_events_to_recording_controls(vr_teleop.get_vr_events()),
                )
                _apply_vr_recording_decision(recording, vr_decision)
                if vr_decision.quit_session:
                    break

            obs = _get_robot_observation(robot, use_camera=False)
            if vr_decision.reset_robot:
                action = vr_teleop.move_to_zero_position(robot)
            else:
                action = vr_teleop.get_action(obs, robot)
            if action:
                sent_action = robot.send_action(action)
            else:
                sent_action = {}
            obs = _get_robot_observation(robot, use_camera=True)
            _publish_vr_camera_frames(obs)
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        _stop_orbbec_rgb_sidecar(orbbec_process)
        try:
            robot.disconnect()
        finally:
            try:
                vr_teleop.disconnect()
            except Exception:
                pass
            if hotkeys.is_connected:
                hotkeys.disconnect()
    return 0


def _create_dataset(
    robot: Any,
    *,
    dataset_id: str,
    dataset_root: str,
    fps: int,
    use_videos: bool,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}
    return LeRobotDataset.create(
        dataset_id,
        fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=use_videos,
    )


def _record_frame_if_needed(recording: RecordingSession | None, observation: dict[str, Any], action: dict[str, Any]) -> None:
    if recording is None or not recording.active:
        return

    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    observation_frame = build_dataset_frame(recording.dataset.features, observation, prefix=OBS_STR)
    complete_action = dict(action)
    for action_name in recording.dataset.features[ACTION]["names"]:
        if action_name in complete_action:
            continue
        if action_name in observation:
            complete_action[action_name] = observation[action_name]
        elif action_name.endswith(".vel"):
            complete_action[action_name] = 0.0
        else:
            complete_action[action_name] = 0.0

    action_frame = build_dataset_frame(recording.dataset.features, complete_action, prefix=ACTION)
    frame = {
        **observation_frame,
        **action_frame,
        "task": recording.task,
        "timestamp": time.time(),
    }
    recording.dataset.add_frame(frame)


def _handle_recording_hotkeys(
    recording: RecordingSession | None,
    pressed_keys: set[str],
    *,
    start_key: str,
    stop_key: str,
) -> None:
    if recording is None:
        return

    if start_key in pressed_keys and not recording.active:
        recording.active = True
        print(f"Recording started. Press `{stop_key}` to save the current episode.")
        return

    if stop_key in pressed_keys and recording.active:
        _save_episode(recording)


def _save_episode(recording: RecordingSession) -> None:
    if not _episode_buffer_has_frames(recording.dataset):
        recording.active = False
        print("Recording stopped. No frames captured, skipping save.")
        return

    recording.dataset.save_episode()
    recording.active = False
    print(f"Saved episode {recording.dataset.meta.total_episodes - 1}.")


def _finalize_recording(recording: RecordingSession | None) -> None:
    if recording is None or not recording.active:
        return
    print("Saving the active episode before exit.")
    _save_episode(recording)


def _discard_episode(recording: RecordingSession) -> None:
    clear_episode_buffer = getattr(recording.dataset, "clear_episode_buffer", None)
    if callable(clear_episode_buffer):
        clear_episode_buffer()
    else:
        buffer = getattr(recording.dataset, "episode_buffer", None)
        if isinstance(buffer, dict):
            for key, value in buffer.items():
                if key == "size":
                    buffer[key] = 0
                elif hasattr(value, "clear"):
                    value.clear()
    recording.active = False
    print("Discarded the current episode.")


def _episode_buffer_has_frames(dataset: Any) -> bool:
    buffer = getattr(dataset, "episode_buffer", None)
    return bool(buffer and buffer.get("size", 0) > 0)


def _print_recording_guide(
    recording: RecordingSession | None,
    *,
    start_key: str,
    stop_key: str,
    quit_key: str,
    controller: str = "keyboard",
) -> None:
    print(f"Quit key: `{quit_key}`")
    if controller == "vr":
        print(
            "VR controls: left thumbstick right start/stop and save, "
            "left thumbstick left discard, left thumbstick up save and quit, "
            "left thumbstick down reset robot pose"
        )
    if recording is None:
        return
    if controller == "keyboard":
        print(f"Recording hotkeys: `{start_key}` start, `{stop_key}` stop and save")


def _load_example_module(file_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to build import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_vr_events_to_recording_controls(vr_events: dict[str, bool] | None) -> VRRecordingControls:
    if not vr_events:
        return VRRecordingControls()

    return VRRecordingControls(
        toggle_recording=bool(vr_events.get("exit_early")),
        discard_episode=bool(vr_events.get("rerecord_episode")),
        quit_session=bool(vr_events.get("stop_recording")),
        reset_robot=bool(vr_events.get("reset_position")),
    )


def _decide_vr_recording_action(active: bool, controls: VRRecordingControls) -> VRRecordingDecision:
    toggle_requested = (
        controls.toggle_recording
        and not controls.discard_episode
        and not controls.quit_session
    )
    return VRRecordingDecision(
        start_recording=toggle_requested and not active,
        save_episode=toggle_requested and active,
        discard_episode=controls.discard_episode and active,
        quit_session=controls.quit_session,
        reset_robot=controls.reset_robot,
    )


def _apply_vr_recording_decision(
    recording: RecordingSession,
    decision: VRRecordingDecision,
) -> None:
    if decision.start_recording:
        recording.active = True
        print("Recording started from VR. Push left thumbstick right again to save.")
    if decision.save_episode:
        _save_episode(recording)
    if decision.discard_episode:
        _discard_episode(recording)
    if decision.quit_session:
        print("Stopping the VR recording session.")


if __name__ == "__main__":
    raise SystemExit(main())
