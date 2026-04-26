from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any
from urllib import request
from urllib.parse import urljoin

from xlerobot_playground import ros_forward_accel_diagnostic

IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32
except Exception as exc:  # pragma: no cover - runtime guard for non-ROS test envs.
    IMPORT_ERROR = exc
    rclpy = None
    Node = object
    Float32 = None


def require_ros() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError("Pitched forward diagnostic requires ROS 2 Python packages.") from IMPORT_ERROR


def post_camera_pitch(
    *,
    robot_brain_url: str,
    pitch_deg: float,
    action_key: str | None,
    settle_s: float,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"pitch_deg": float(pitch_deg), "settle_s": float(settle_s)}
    if action_key:
        payload["action_key"] = action_key
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        urljoin(robot_brain_url.rstrip("/") + "/", "camera/head/pitch"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    effective_timeout_s = max(float(timeout_s), float(settle_s) + 5.0)
    try:
        with request.urlopen(req, timeout=effective_timeout_s) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Failed to command physical camera pitch through robot brain. "
            "Make sure robot_brain_agent is restarted with --allow-motion-commands and "
            "--camera-pitch-action-key set to the real head tilt action key."
        ) from exc
    return json.loads(data.decode("utf-8")) if data else {}


class PitchWaitNode(Node):
    def __init__(self, *, topic: str) -> None:
        super().__init__("xlerobot_pitched_forward_diagnostic")
        self.pitch_rad: float | None = None
        self.create_subscription(Float32, topic, self._on_pitch, 10)

    def _on_pitch(self, message: Any) -> None:
        self.pitch_rad = float(message.data)


def wait_for_pitch_topic(
    *,
    topic: str,
    expected_pitch_deg: float,
    tolerance_deg: float,
    timeout_s: float,
) -> float:
    require_ros()
    rclpy.init()
    node = PitchWaitNode(topic=topic)
    deadline_s = time.monotonic() + max(float(timeout_s), 0.0)
    expected_rad = math.radians(float(expected_pitch_deg))
    tolerance_rad = math.radians(abs(float(tolerance_deg)))
    try:
        while time.monotonic() < deadline_s:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.pitch_rad is not None and abs(node.pitch_rad - expected_rad) <= tolerance_rad:
                return node.pitch_rad
        last = "none" if node.pitch_rad is None else f"{math.degrees(node.pitch_rad):.2f}deg"
        raise RuntimeError(
            f"Timed out waiting for {topic} to reach {expected_pitch_deg:.2f}deg "
            f"(tolerance {tolerance_deg:.2f}deg, last={last})."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_forward_args(args: argparse.Namespace) -> list[str]:
    return [
        "--send-motion",
        "--duration-s",
        str(args.duration_s),
        "--sample-hz",
        str(args.sample_hz),
        "--linear-m-s",
        str(args.linear_m_s),
        "--target-distance-m",
        str(args.target_distance_m),
        "--target-source",
        args.target_source,
        "--imu-topic",
        args.imu_topic,
        "--imu-frame-convention",
        args.imu_frame_convention,
        "--accel-bias-calibration-s",
        str(args.accel_bias_calibration_s),
        "--max-imu-staleness-s",
        str(args.max_imu_staleness_s),
        "--csv-out",
        args.csv_out,
        "--json-out",
        args.json_out,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Physically pitch the camera through robot_brain_agent, verify ROS receives it, "
            "then run a 1m forward RGB-D odometry diagnostic."
        )
    )
    parser.add_argument("--robot-brain-url", default="http://127.0.0.1:8765")
    parser.add_argument("--camera-pitch-deg", type=float, default=35.0)
    parser.add_argument(
        "--camera-pitch-action-key",
        default=None,
        help="Optional robot action key override for head pitch. Otherwise robot_brain_agent must be configured with one.",
    )
    parser.add_argument("--camera-pitch-settle-s", type=float, default=2.0)
    parser.add_argument("--camera-pitch-topic", default="/camera/head/pitch_rad")
    parser.add_argument("--pitch-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--pitch-wait-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--no-physical-camera-pitch",
        action="store_true",
        help="Skip the physical pitch command; only wait for the ROS topic and run the forward test.",
    )
    parser.add_argument(
        "--no-wait-for-ros-pitch",
        action="store_true",
        help="Do not verify /camera/head/pitch_rad before running the forward test.",
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sample-hz", type=float, default=30.0)
    parser.add_argument("--linear-m-s", type=float, default=0.03)
    parser.add_argument("--target-distance-m", type=float, default=1.0)
    parser.add_argument("--target-source", choices=("tf", "odom", "accelerometer"), default="tf")
    parser.add_argument("--imu-topic", default="/imu/filtered_yaw")
    parser.add_argument("--imu-frame-convention", choices=("camera_optical", "base_link"), default="base_link")
    parser.add_argument("--accel-bias-calibration-s", type=float, default=0.0)
    parser.add_argument("--max-imu-staleness-s", type=float, default=0.5)
    parser.add_argument("--csv-out", default="artifacts/diagnostics/forward_pitch35_1m.csv")
    parser.add_argument("--json-out", default="artifacts/diagnostics/forward_pitch35_1m_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.no_physical_camera_pitch:
        response = post_camera_pitch(
            robot_brain_url=args.robot_brain_url,
            pitch_deg=args.camera_pitch_deg,
            action_key=args.camera_pitch_action_key,
            settle_s=args.camera_pitch_settle_s,
        )
        camera = response.get("metadata", {}).get("camera", {})
        print(
            "[pitched_forward_diag] physical camera pitch command completed: "
            f"pitch_deg={float(camera.get('pitch_deg', args.camera_pitch_deg)):.2f}",
            flush=True,
        )
    if not args.no_wait_for_ros_pitch:
        pitch_rad = wait_for_pitch_topic(
            topic=args.camera_pitch_topic,
            expected_pitch_deg=args.camera_pitch_deg,
            tolerance_deg=args.pitch_tolerance_deg,
            timeout_s=args.pitch_wait_timeout_s,
        )
        print(
            "[pitched_forward_diag] ROS camera pitch confirmed: "
            f"{math.degrees(pitch_rad):.2f}deg on {args.camera_pitch_topic}",
            flush=True,
        )
    return ros_forward_accel_diagnostic.main(build_forward_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
