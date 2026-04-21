from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time as RosTime
    from sensor_msgs.msg import Imu
    from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
except Exception as exc:  # pragma: no cover - runtime guard for non-ROS test envs.
    IMPORT_ERROR: Exception | None = exc
    rclpy = None
    Twist = None
    Odometry = None
    Imu = None
    Node = object
    RosTime = None
    Buffer = None
    TransformListener = None
    ConnectivityException = Exception
    ExtrapolationException = Exception
    LookupException = Exception
else:
    IMPORT_ERROR = None


def require_ros() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError("ROS rotation diagnostic requires ROS 2 Python packages.") from IMPORT_ERROR


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def imu_value_from_source(*, source: str, gyro_x: float, gyro_y: float, gyro_z: float) -> float:
    normalized = str(source).strip().lower()
    if normalized == "x":
        return gyro_x
    if normalized == "y":
        return gyro_y
    if normalized == "z":
        return gyro_z
    if normalized in {"robot_yaw", "optical_yaw"}:
        # Gemini 2 publishes raw sensor data in camera optical coordinates:
        # optical X right, Y down, Z forward. For an aligned robot body,
        # base_link yaw rate (around Z up) maps to -optical Y.
        return -gyro_y
    raise ValueError(f"Unsupported IMU source: {source}")


class RotationDiagnosticNode(Node):
    def __init__(
        self,
        *,
        odom_frame: str,
        base_frame: str,
        cmd_vel_topic: str,
        odom_topic: str,
        imu_topic: str,
        imu_axis: str,
        sample_hz: float,
    ) -> None:
        super().__init__("xlerobot_rotation_diagnostic")
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.imu_axis = str(imu_axis).strip().lower()
        self.sample_dt_s = 1.0 / max(float(sample_hz), 1e-6)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.latest_odom_pose: dict[str, float] | None = None
        self.latest_imu_sample: dict[str, float] | None = None
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 50)

    def _on_odom(self, message: Any) -> None:
        position = message.pose.pose.position
        rotation = message.pose.pose.orientation
        self.latest_odom_pose = {
            "x": float(position.x),
            "y": float(position.y),
            "yaw_rad": yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        }

    def _on_imu(self, message: Any) -> None:
        stamp = getattr(message, "header", None)
        stamp_s = time.time()
        if stamp is not None:
            stamp_s = float(getattr(stamp.stamp, "sec", 0)) + float(getattr(stamp.stamp, "nanosec", 0)) / 1_000_000_000.0
        self.latest_imu_sample = {
            "timestamp_s": stamp_s,
            "angular_velocity_x_rad_s": float(message.angular_velocity.x),
            "angular_velocity_y_rad_s": float(message.angular_velocity.y),
            "angular_velocity_z_rad_s": float(message.angular_velocity.z),
            "linear_acceleration_x_m_s2": float(message.linear_acceleration.x),
            "linear_acceleration_y_m_s2": float(message.linear_acceleration.y),
            "linear_acceleration_z_m_s2": float(message.linear_acceleration.z),
        }

    def lookup_pose(self) -> dict[str, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, RosTime())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "x": float(translation.x),
            "y": float(translation.y),
            "yaw_rad": yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        }

    def publish_spin(self, angular_rad_s: float) -> None:
        twist = Twist()
        twist.angular.z = float(angular_rad_s)
        self.cmd_vel_pub.publish(twist)

    def stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def collect(
        self,
        *,
        duration_s: float,
        angular_rad_s: float,
        send_motion: bool,
        target_yaw_rad: float | None = None,
        target_source: str = "tf",
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        start = time.time()
        deadline = start + max(float(duration_s), 0.0)
        target_abs_yaw_rad = abs(float(target_yaw_rad)) if target_yaw_rad is not None else None
        previous_tf_yaw: float | None = None
        previous_odom_yaw: float | None = None
        previous_imu_stamp_s: float | None = None
        unwrapped_tf_yaw = 0.0
        unwrapped_odom_yaw = 0.0
        unwrapped_imu_yaw = 0.0
        stop_reason = "duration_timeout"
        while time.time() < deadline:
            now = time.time()
            if send_motion:
                self.publish_spin(angular_rad_s)
            rclpy.spin_once(self, timeout_sec=0.0)
            pose = self.lookup_pose()
            odom_pose = self.latest_odom_pose
            imu_sample = self.latest_imu_sample
            sample: dict[str, Any] = {
                "t_s": round(now - start, 4),
                "cmd_angular_rad_s": angular_rad_s if send_motion else 0.0,
            }
            if pose is None:
                sample["tf_pose_available"] = False
            else:
                yaw = float(pose["yaw_rad"])
                if previous_tf_yaw is not None:
                    unwrapped_tf_yaw += angle_delta(yaw, previous_tf_yaw)
                previous_tf_yaw = yaw
                sample.update(
                    {
                        "tf_pose_available": True,
                        "tf_x_m": pose["x"],
                        "tf_y_m": pose["y"],
                        "tf_yaw_rad": yaw,
                        "tf_yaw_deg": math.degrees(yaw),
                        "tf_unwrapped_yaw_rad": unwrapped_tf_yaw,
                        "tf_unwrapped_yaw_deg": math.degrees(unwrapped_tf_yaw),
                    }
                )
            if odom_pose is None:
                sample["odom_pose_available"] = False
            else:
                odom_yaw = float(odom_pose["yaw_rad"])
                if previous_odom_yaw is not None:
                    unwrapped_odom_yaw += angle_delta(odom_yaw, previous_odom_yaw)
                previous_odom_yaw = odom_yaw
                sample.update(
                    {
                        "odom_pose_available": True,
                        "odom_x_m": odom_pose["x"],
                        "odom_y_m": odom_pose["y"],
                        "odom_yaw_rad": odom_yaw,
                        "odom_yaw_deg": math.degrees(odom_yaw),
                        "odom_unwrapped_yaw_rad": unwrapped_odom_yaw,
                        "odom_unwrapped_yaw_deg": math.degrees(unwrapped_odom_yaw),
                    }
                )
            if imu_sample is None:
                sample["imu_available"] = False
            else:
                imu_stamp_s = float(imu_sample["timestamp_s"])
                gyro_x = float(imu_sample["angular_velocity_x_rad_s"])
                gyro_y = float(imu_sample["angular_velocity_y_rad_s"])
                gyro_z = float(imu_sample["angular_velocity_z_rad_s"])
                gyro_axis_value = imu_value_from_source(
                    source=self.imu_axis,
                    gyro_x=gyro_x,
                    gyro_y=gyro_y,
                    gyro_z=gyro_z,
                )
                imu_dt_s = self.sample_dt_s if previous_imu_stamp_s is None else max(0.0, imu_stamp_s - previous_imu_stamp_s)
                previous_imu_stamp_s = imu_stamp_s
                unwrapped_imu_yaw += gyro_axis_value * imu_dt_s
                sample.update(
                    {
                        "imu_available": True,
                        "imu_timestamp_s": imu_stamp_s,
                        "imu_dt_s": imu_dt_s,
                        "imu_axis": self.imu_axis,
                        "imu_angular_velocity_x_rad_s": gyro_x,
                        "imu_angular_velocity_y_rad_s": gyro_y,
                        "imu_angular_velocity_z_rad_s": gyro_z,
                        "imu_angular_velocity_axis_rad_s": gyro_axis_value,
                        "imu_linear_acceleration_x_m_s2": float(imu_sample["linear_acceleration_x_m_s2"]),
                        "imu_linear_acceleration_y_m_s2": float(imu_sample["linear_acceleration_y_m_s2"]),
                        "imu_linear_acceleration_z_m_s2": float(imu_sample["linear_acceleration_z_m_s2"]),
                        "imu_unwrapped_yaw_rad": unwrapped_imu_yaw,
                        "imu_unwrapped_yaw_deg": math.degrees(unwrapped_imu_yaw),
                    }
                )
            samples.append(sample)
            if target_abs_yaw_rad is not None:
                unwrapped_key = f"{target_source}_unwrapped_yaw_rad"
                if unwrapped_key in sample and abs(float(sample[unwrapped_key])) >= target_abs_yaw_rad:
                    stop_reason = f"target_{target_source}_yaw_reached"
                    break
            time.sleep(self.sample_dt_s)
        self.stop()
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.stop()
        if samples:
            samples[-1]["stop_reason"] = stop_reason
        return samples


def _summarize_source(samples: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get(f"{prefix}_pose_available")]
    if not valid:
        return {
            "valid_pose_count": 0,
            "message": f"No {prefix} pose samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    elapsed_s = max(float(end["t_s"]) - float(start["t_s"]), 1e-6)
    yaw_delta_rad = float(end[f"{prefix}_unwrapped_yaw_rad"]) - float(start[f"{prefix}_unwrapped_yaw_rad"])
    drift_m = math.hypot(float(end[f"{prefix}_x_m"]) - float(start[f"{prefix}_x_m"]), float(end[f"{prefix}_y_m"]) - float(start[f"{prefix}_y_m"]))
    return {
        "valid_pose_count": len(valid),
        "elapsed_s": round(elapsed_s, 3),
        "unwrapped_yaw_delta_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_deg": round(math.degrees(yaw_delta_rad), 2),
        "mean_yaw_rate_rad_s": round(yaw_delta_rad / elapsed_s, 4),
        "translation_drift_m": round(drift_m, 4),
        "start": {
            "x_m": start[f"{prefix}_x_m"],
            "y_m": start[f"{prefix}_y_m"],
            "yaw_deg": start[f"{prefix}_yaw_deg"],
        },
        "end": {
            "x_m": end[f"{prefix}_x_m"],
            "y_m": end[f"{prefix}_y_m"],
            "yaw_deg": end[f"{prefix}_yaw_deg"],
        },
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "stop_reason": samples[-1].get("stop_reason") if samples else "no_samples",
        "tf": _summarize_source(samples, prefix="tf"),
        "odom_topic": _summarize_source(samples, prefix="odom"),
        "imu": _summarize_imu(samples),
    }


def _summarize_imu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("imu_available")]
    if not valid:
        return {
            "valid_sample_count": 0,
            "message": "No IMU samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    elapsed_s = max(float(end["t_s"]) - float(start["t_s"]), 1e-6)
    yaw_delta_rad = float(end["imu_unwrapped_yaw_rad"]) - float(start["imu_unwrapped_yaw_rad"])
    yaw_delta_deg = float(end["imu_unwrapped_yaw_deg"]) - float(start["imu_unwrapped_yaw_deg"])
    return {
        "valid_sample_count": len(valid),
        "elapsed_s": round(elapsed_s, 3),
        "reported_turn_deg": round(yaw_delta_deg, 2),
        "reported_turn_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_rad": round(yaw_delta_rad, 4),
        "unwrapped_yaw_delta_deg": round(yaw_delta_deg, 2),
        "mean_yaw_rate_rad_s": round(yaw_delta_rad / elapsed_s, 4),
        "start": {
            "yaw_deg": start["imu_unwrapped_yaw_deg"],
            "axis": start["imu_axis"],
            "angular_velocity_axis_rad_s": start["imu_angular_velocity_axis_rad_s"],
            "angular_velocity_x_rad_s": start["imu_angular_velocity_x_rad_s"],
            "angular_velocity_y_rad_s": start["imu_angular_velocity_y_rad_s"],
            "angular_velocity_z_rad_s": start["imu_angular_velocity_z_rad_s"],
        },
        "end": {
            "yaw_deg": end["imu_unwrapped_yaw_deg"],
            "axis": end["imu_axis"],
            "angular_velocity_axis_rad_s": end["imu_angular_velocity_axis_rad_s"],
            "angular_velocity_x_rad_s": end["imu_angular_velocity_x_rad_s"],
            "angular_velocity_y_rad_s": end["imu_angular_velocity_y_rad_s"],
            "angular_velocity_z_rad_s": end["imu_angular_velocity_z_rad_s"],
        },
    }


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for sample in samples for key in sample})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record odometry yaw during passive or commanded in-place rotation.")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument(
        "--imu-axis",
        choices=("x", "y", "z", "robot_yaw", "optical_yaw"),
        default="robot_yaw",
        help="Raw IMU axis or derived robot yaw source. For an aligned Gemini 2 camera, use robot_yaw.",
    )
    parser.add_argument("--angular-rad-s", type=float, default=0.10)
    parser.add_argument(
        "--target-yaw-deg",
        type=float,
        default=None,
        help="Stop early when the selected pose source reports this absolute yaw delta. --duration-s remains the safety timeout.",
    )
    parser.add_argument(
        "--target-source",
        choices=("tf", "odom", "imu"),
        default="tf",
        help="Source used by --target-yaw-deg. Use 'tf' for odom->base_link, 'odom' for /odom, or 'imu' for integrated selected IMU source.",
    )
    parser.add_argument(
        "--send-motion",
        action="store_true",
        help="Actually publish angular /cmd_vel. Without this flag the script only records TF.",
    )
    parser.add_argument("--csv-out", default="artifacts/diagnostics/rotation_diagnostic.csv")
    parser.add_argument("--json-out", default="artifacts/diagnostics/rotation_diagnostic_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros()
    rclpy.init()
    node = RotationDiagnosticNode(
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        imu_axis=args.imu_axis,
        sample_hz=args.sample_hz,
    )
    try:
        samples = node.collect(
            duration_s=args.duration_s,
            angular_rad_s=args.angular_rad_s,
            send_motion=args.send_motion,
            target_yaw_rad=math.radians(args.target_yaw_deg) if args.target_yaw_deg is not None else None,
            target_source=args.target_source,
        )
        summary = summarize(samples)
        write_csv(Path(args.csv_out).expanduser(), samples)
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        imu_reported_turn_deg = summary.get("imu", {}).get("reported_turn_deg")
        if imu_reported_turn_deg is not None:
            print(f"Reported IMU turn: {imu_reported_turn_deg} deg")
        print(f"Wrote samples: {Path(args.csv_out).expanduser()}")
        print(f"Wrote summary: {json_path}")
        return 0 if (
            summary.get("tf", {}).get("valid_pose_count", 0)
            or summary.get("odom_topic", {}).get("valid_pose_count", 0)
            or summary.get("imu", {}).get("valid_sample_count", 0)
        ) else 2
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
