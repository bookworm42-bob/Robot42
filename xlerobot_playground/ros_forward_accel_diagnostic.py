from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from xlerobot_playground.rgbd_visual_odometry import imu_to_base_components

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
        raise RuntimeError("ROS forward accelerometer diagnostic requires ROS 2 Python packages.") from IMPORT_ERROR


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def integrate_acceleration_step(
    *,
    distance_m: float,
    velocity_m_s: float,
    acceleration_m_s2: float,
    dt_s: float,
    acceleration_deadband_m_s2: float,
    velocity_damping_per_s: float,
    max_velocity_m_s: float,
) -> tuple[float, float, float]:
    dt_s = max(float(dt_s), 0.0)
    if dt_s <= 0.0:
        return float(distance_m), float(velocity_m_s), 0.0
    corrected_accel = float(acceleration_m_s2)
    if abs(corrected_accel) < max(float(acceleration_deadband_m_s2), 0.0):
        corrected_accel = 0.0
    damping = max(0.0, 1.0 - max(float(velocity_damping_per_s), 0.0) * dt_s)
    next_velocity = (float(velocity_m_s) + corrected_accel * dt_s) * damping
    max_velocity_m_s = max(float(max_velocity_m_s), 1e-6)
    next_velocity = clamp(next_velocity, -max_velocity_m_s, max_velocity_m_s)
    next_distance = float(distance_m) + (float(velocity_m_s) + next_velocity) * 0.5 * dt_s
    return next_distance, next_velocity, corrected_accel


def forward_displacement_m(
    *,
    start_x_m: float,
    start_y_m: float,
    start_yaw_rad: float,
    current_x_m: float,
    current_y_m: float,
) -> tuple[float, float]:
    dx = float(current_x_m) - float(start_x_m)
    dy = float(current_y_m) - float(start_y_m)
    cos_yaw = math.cos(float(start_yaw_rad))
    sin_yaw = math.sin(float(start_yaw_rad))
    forward_m = dx * cos_yaw + dy * sin_yaw
    lateral_m = -dx * sin_yaw + dy * cos_yaw
    return forward_m, lateral_m


class ForwardAccelDiagnosticNode(Node):
    def __init__(
        self,
        *,
        odom_frame: str,
        base_frame: str,
        cmd_vel_topic: str,
        odom_topic: str,
        imu_topic: str,
        imu_frame_convention: str,
        sample_hz: float,
        accel_bias_calibration_s: float,
        accel_bias_min_samples: int,
        acceleration_deadband_m_s2: float,
        velocity_damping_per_s: float,
        max_estimated_velocity_m_s: float,
    ) -> None:
        super().__init__("xlerobot_forward_accel_diagnostic")
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.imu_frame_convention = str(imu_frame_convention).strip().lower()
        self.sample_dt_s = 1.0 / max(float(sample_hz), 1e-6)
        self.accel_bias_calibration_s = max(float(accel_bias_calibration_s), 0.0)
        self.accel_bias_min_samples = max(int(accel_bias_min_samples), 1)
        self.acceleration_deadband_m_s2 = max(float(acceleration_deadband_m_s2), 0.0)
        self.velocity_damping_per_s = max(float(velocity_damping_per_s), 0.0)
        self.max_estimated_velocity_m_s = max(float(max_estimated_velocity_m_s), 1e-6)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.latest_odom_pose: dict[str, float] | None = None
        self.latest_imu_sample: dict[str, float] | None = None
        self.accel_bias_forward_m_s2 = 0.0
        self.accel_bias_lateral_m_s2 = 0.0
        self.accel_bias_vertical_m_s2 = 0.0
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 200)

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
        linear_x = float(message.linear_acceleration.x)
        linear_y = float(message.linear_acceleration.y)
        linear_z = float(message.linear_acceleration.z)
        base_x, base_y, base_z = imu_to_base_components(
            frame_convention=self.imu_frame_convention,
            x=linear_x,
            y=linear_y,
            z=linear_z,
        )
        self.latest_imu_sample = {
            "timestamp_s": stamp_s,
            "linear_acceleration_x_m_s2": linear_x,
            "linear_acceleration_y_m_s2": linear_y,
            "linear_acceleration_z_m_s2": linear_z,
            "base_linear_acceleration_x_m_s2": float(base_x),
            "base_linear_acceleration_y_m_s2": float(base_y),
            "base_linear_acceleration_z_m_s2": float(base_z),
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

    def publish_forward(self, linear_m_s: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear_m_s)
        self.cmd_vel_pub.publish(twist)

    def stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def calibrate_accel_bias(self) -> dict[str, float] | None:
        if self.accel_bias_calibration_s <= 1e-6:
            return {
                "bias_forward_m_s2": self.accel_bias_forward_m_s2,
                "bias_lateral_m_s2": self.accel_bias_lateral_m_s2,
                "bias_vertical_m_s2": self.accel_bias_vertical_m_s2,
                "sample_count": 0,
                "elapsed_s": 0.0,
            }
        start = time.time()
        deadline = start + self.accel_bias_calibration_s
        sum_forward = 0.0
        sum_lateral = 0.0
        sum_vertical = 0.0
        count = 0
        last_timestamp_s: float | None = None
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(self.sample_dt_s, 0.05))
            imu_sample = self.latest_imu_sample
            if imu_sample is None:
                continue
            timestamp_s = float(imu_sample["timestamp_s"])
            if last_timestamp_s is not None and timestamp_s <= last_timestamp_s:
                continue
            last_timestamp_s = timestamp_s
            sum_forward += float(imu_sample["base_linear_acceleration_x_m_s2"])
            sum_lateral += float(imu_sample["base_linear_acceleration_y_m_s2"])
            sum_vertical += float(imu_sample["base_linear_acceleration_z_m_s2"])
            count += 1
        if count < self.accel_bias_min_samples:
            return None
        self.accel_bias_forward_m_s2 = sum_forward / count
        self.accel_bias_lateral_m_s2 = sum_lateral / count
        self.accel_bias_vertical_m_s2 = sum_vertical / count
        return {
            "bias_forward_m_s2": self.accel_bias_forward_m_s2,
            "bias_lateral_m_s2": self.accel_bias_lateral_m_s2,
            "bias_vertical_m_s2": self.accel_bias_vertical_m_s2,
            "sample_count": count,
            "elapsed_s": round(time.time() - start, 3),
        }

    def collect(
        self,
        *,
        duration_s: float,
        linear_m_s: float,
        send_motion: bool,
        target_distance_m: float | None = None,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        accel_bias = self.calibrate_accel_bias()
        start = time.time()
        deadline = start + max(float(duration_s), 0.0)
        target_abs_distance_m = abs(float(target_distance_m)) if target_distance_m is not None else None
        previous_imu_stamp_s: float | None = None
        estimated_distance_m = 0.0
        estimated_velocity_m_s = 0.0
        tf_start_pose: dict[str, float] | None = None
        odom_start_pose: dict[str, float] | None = None
        stop_reason = "duration_timeout"
        while time.time() < deadline:
            now = time.time()
            rclpy.spin_once(self, timeout_sec=0.0)
            pose = self.lookup_pose()
            odom_pose = self.latest_odom_pose
            imu_sample = self.latest_imu_sample
            if tf_start_pose is None and pose is not None:
                tf_start_pose = dict(pose)
            if odom_start_pose is None and odom_pose is not None:
                odom_start_pose = dict(odom_pose)
            sample: dict[str, Any] = {
                "t_s": round(now - start, 4),
            }
            if accel_bias is not None:
                sample.update(
                    {
                        "accel_bias_forward_m_s2": accel_bias["bias_forward_m_s2"],
                        "accel_bias_lateral_m_s2": accel_bias["bias_lateral_m_s2"],
                        "accel_bias_vertical_m_s2": accel_bias["bias_vertical_m_s2"],
                    }
                )
            if pose is None:
                sample["tf_pose_available"] = False
            else:
                sample.update(
                    {
                        "tf_pose_available": True,
                        "tf_x_m": float(pose["x"]),
                        "tf_y_m": float(pose["y"]),
                        "tf_yaw_rad": float(pose["yaw_rad"]),
                    }
                )
                if tf_start_pose is not None:
                    tf_forward_m, tf_lateral_m = forward_displacement_m(
                        start_x_m=tf_start_pose["x"],
                        start_y_m=tf_start_pose["y"],
                        start_yaw_rad=tf_start_pose["yaw_rad"],
                        current_x_m=pose["x"],
                        current_y_m=pose["y"],
                    )
                    sample["tf_forward_distance_m"] = tf_forward_m
                    sample["tf_lateral_distance_m"] = tf_lateral_m
            if odom_pose is None:
                sample["odom_pose_available"] = False
            else:
                sample.update(
                    {
                        "odom_pose_available": True,
                        "odom_x_m": float(odom_pose["x"]),
                        "odom_y_m": float(odom_pose["y"]),
                        "odom_yaw_rad": float(odom_pose["yaw_rad"]),
                    }
                )
                if odom_start_pose is not None:
                    odom_forward_m, odom_lateral_m = forward_displacement_m(
                        start_x_m=odom_start_pose["x"],
                        start_y_m=odom_start_pose["y"],
                        start_yaw_rad=odom_start_pose["yaw_rad"],
                        current_x_m=odom_pose["x"],
                        current_y_m=odom_pose["y"],
                    )
                    sample["odom_forward_distance_m"] = odom_forward_m
                    sample["odom_lateral_distance_m"] = odom_lateral_m
            if imu_sample is None:
                sample["imu_available"] = False
            else:
                imu_stamp_s = float(imu_sample["timestamp_s"])
                dt_s = self.sample_dt_s if previous_imu_stamp_s is None else max(0.0, imu_stamp_s - previous_imu_stamp_s)
                previous_imu_stamp_s = imu_stamp_s
                raw_forward_m_s2 = float(imu_sample["base_linear_acceleration_x_m_s2"])
                raw_lateral_m_s2 = float(imu_sample["base_linear_acceleration_y_m_s2"])
                raw_vertical_m_s2 = float(imu_sample["base_linear_acceleration_z_m_s2"])
                corrected_forward_m_s2 = raw_forward_m_s2 - self.accel_bias_forward_m_s2
                estimated_distance_m, estimated_velocity_m_s, used_forward_m_s2 = integrate_acceleration_step(
                    distance_m=estimated_distance_m,
                    velocity_m_s=estimated_velocity_m_s,
                    acceleration_m_s2=corrected_forward_m_s2,
                    dt_s=dt_s,
                    acceleration_deadband_m_s2=self.acceleration_deadband_m_s2,
                    velocity_damping_per_s=self.velocity_damping_per_s,
                    max_velocity_m_s=self.max_estimated_velocity_m_s,
                )
                sample.update(
                    {
                        "imu_available": True,
                        "imu_timestamp_s": imu_stamp_s,
                        "imu_dt_s": dt_s,
                        "imu_raw_linear_acceleration_x_m_s2": float(imu_sample["linear_acceleration_x_m_s2"]),
                        "imu_raw_linear_acceleration_y_m_s2": float(imu_sample["linear_acceleration_y_m_s2"]),
                        "imu_raw_linear_acceleration_z_m_s2": float(imu_sample["linear_acceleration_z_m_s2"]),
                        "imu_base_linear_acceleration_x_m_s2": raw_forward_m_s2,
                        "imu_base_linear_acceleration_y_m_s2": raw_lateral_m_s2,
                        "imu_base_linear_acceleration_z_m_s2": raw_vertical_m_s2,
                        "imu_corrected_forward_acceleration_m_s2": corrected_forward_m_s2,
                        "imu_used_forward_acceleration_m_s2": used_forward_m_s2,
                        "imu_estimated_forward_velocity_m_s": estimated_velocity_m_s,
                        "imu_estimated_forward_distance_m": estimated_distance_m,
                    }
                )
            command_linear_m_s = 0.0
            target_reached = False
            if send_motion:
                command_linear_m_s = float(linear_m_s)
                if target_abs_distance_m is not None:
                    remaining_m = max(target_abs_distance_m - abs(estimated_distance_m), 0.0)
                    sample["target_remaining_distance_m"] = remaining_m
                    if remaining_m <= 0.0:
                        command_linear_m_s = 0.0
                        target_reached = True
                self.publish_forward(command_linear_m_s)
            sample["cmd_linear_m_s"] = command_linear_m_s
            samples.append(sample)
            if target_abs_distance_m is not None and target_reached:
                stop_reason = "target_accel_distance_reached"
                break
            time.sleep(self.sample_dt_s)
        self.stop()
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.stop()
        if samples:
            samples[-1]["stop_reason"] = stop_reason
        return samples


def _summarize_pose_source(samples: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    distance_key = f"{prefix}_forward_distance_m"
    lateral_key = f"{prefix}_lateral_distance_m"
    valid = [sample for sample in samples if distance_key in sample]
    if not valid:
        return {
            "valid_pose_count": 0,
            "message": f"No {prefix} pose samples were available.",
        }
    start = valid[0]
    end = valid[-1]
    return {
        "valid_pose_count": len(valid),
        "elapsed_s": round(max(float(end["t_s"]) - float(start["t_s"]), 1e-6), 3),
        "forward_distance_m": round(float(end[distance_key]), 4),
        "lateral_distance_m": round(float(end[lateral_key]), 4),
    }


def _summarize_accel(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("imu_available")]
    if not valid:
        return {
            "valid_sample_count": 0,
            "message": "No IMU samples were available.",
        }
    end = valid[-1]
    peak_used_accel = max(abs(float(sample.get("imu_used_forward_acceleration_m_s2", 0.0))) for sample in valid)
    peak_velocity = max(abs(float(sample.get("imu_estimated_forward_velocity_m_s", 0.0))) for sample in valid)
    return {
        "valid_sample_count": len(valid),
        "elapsed_s": round(max(float(end["t_s"]) - float(valid[0]["t_s"]), 1e-6), 3),
        "reported_distance_m": round(float(end.get("imu_estimated_forward_distance_m", 0.0)), 4),
        "reported_velocity_m_s": round(float(end.get("imu_estimated_forward_velocity_m_s", 0.0)), 4),
        "peak_used_forward_acceleration_m_s2": round(peak_used_accel, 4),
        "peak_estimated_velocity_m_s": round(peak_velocity, 4),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "sample_count": len(samples),
        "stop_reason": samples[-1].get("stop_reason") if samples else "no_samples",
        "accelerometer": _summarize_accel(samples),
        "tf": _summarize_pose_source(samples, prefix="tf"),
        "odom_topic": _summarize_pose_source(samples, prefix="odom"),
    }
    if samples and "accel_bias_forward_m_s2" in samples[0]:
        summary["accelerometer_bias"] = {
            "forward_m_s2": round(float(samples[0]["accel_bias_forward_m_s2"]), 6),
            "lateral_m_s2": round(float(samples[0]["accel_bias_lateral_m_s2"]), 6),
            "vertical_m_s2": round(float(samples[0]["accel_bias_vertical_m_s2"]), 6),
        }
    return summary


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for sample in samples for key in sample})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive forward and estimate distance from raw accelerometer integration while logging TF and /odom for comparison."
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--accel-bias-calibration-s", type=float, default=1.0)
    parser.add_argument("--accel-bias-min-samples", type=int, default=50)
    parser.add_argument("--acceleration-deadband-m-s2", type=float, default=0.08)
    parser.add_argument("--velocity-damping-per-s", type=float, default=0.35)
    parser.add_argument("--max-estimated-velocity-m-s", type=float, default=0.75)
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument(
        "--imu-frame-convention",
        choices=("camera_optical", "base_link"),
        default="camera_optical",
        help="Interpret raw IMU vectors in camera optical coordinates or already-rotated base_link coordinates.",
    )
    parser.add_argument("--linear-m-s", type=float, default=0.03)
    parser.add_argument(
        "--target-distance-m",
        type=float,
        default=0.45,
        help="Stop early when the accelerometer-only integrated forward distance reaches this absolute distance.",
    )
    parser.add_argument(
        "--send-motion",
        action="store_true",
        help="Actually publish forward /cmd_vel. Without this flag the script only records accelerometer, TF, and /odom.",
    )
    parser.add_argument("--csv-out", default="artifacts/diagnostics/forward_accel_diagnostic.csv")
    parser.add_argument("--json-out", default="artifacts/diagnostics/forward_accel_diagnostic_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_ros()
    rclpy.init()
    node = ForwardAccelDiagnosticNode(
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        imu_frame_convention=args.imu_frame_convention,
        sample_hz=args.sample_hz,
        accel_bias_calibration_s=args.accel_bias_calibration_s,
        accel_bias_min_samples=args.accel_bias_min_samples,
        acceleration_deadband_m_s2=args.acceleration_deadband_m_s2,
        velocity_damping_per_s=args.velocity_damping_per_s,
        max_estimated_velocity_m_s=args.max_estimated_velocity_m_s,
    )
    try:
        samples = node.collect(
            duration_s=args.duration_s,
            linear_m_s=args.linear_m_s,
            send_motion=args.send_motion,
            target_distance_m=args.target_distance_m,
        )
        summary = summarize(samples)
        write_csv(Path(args.csv_out).expanduser(), samples)
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        accel_distance_m = summary.get("accelerometer", {}).get("reported_distance_m")
        if accel_distance_m is not None:
            print(f"Reported accelerometer-only distance: {accel_distance_m} m")
        print(f"Wrote samples: {Path(args.csv_out).expanduser()}")
        print(f"Wrote summary: {json_path}")
        return 0 if (
            summary.get("accelerometer", {}).get("valid_sample_count", 0)
            or summary.get("tf", {}).get("valid_pose_count", 0)
            or summary.get("odom_topic", {}).get("valid_pose_count", 0)
        ) else 2
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
