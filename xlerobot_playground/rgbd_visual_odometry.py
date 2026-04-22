from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any

IMPORT_ERROR: Exception | None = None
CV_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime guard.
    CV_IMPORT_ERROR = exc
    np = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime guard.
    CV_IMPORT_ERROR = exc
    cv2 = None

try:
    import rclpy
    from geometry_msgs.msg import Quaternion, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from tf2_ros import TransformBroadcaster
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    Quaternion = None
    TransformStamped = None
    Odometry = None
    Node = object
    CameraInfo = None
    Image = None
    Imu = None
    TransformBroadcaster = None


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbdVoConfig:
    rgb_topic: str = "/camera/head/image_raw"
    depth_topic: str = "/camera/head/depth/image_raw"
    camera_info_topic: str = "/camera/head/camera_info"
    imu_topic: str = "/imu"
    odom_topic: str = "/odom"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "head_camera_link"
    publish_rate_hz: float = 15.0
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    min_matches: int = 20
    min_inliers: int = 12
    max_translation_step_m: float = 0.25
    max_yaw_step_rad: float = math.radians(30.0)
    imu_velocity_damping_per_s: float = 0.35
    imu_max_planar_speed_m_s: float = 0.75
    vo_translation_weight: float = 0.85
    imu_stale_after_s: float = 0.5
    imu_frame_convention: str = "camera_optical"
    imu_bias_calibration_s: float = 2.0
    imu_bias_min_samples: int = 20
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0
    camera_yaw_rad: float = 0.0


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class VisualOdomFrame:
    gray: Any
    depth_m: Any
    stamp: Any
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class VisualOdomEstimate:
    translation_m: float
    yaw_rad: float
    matches: int
    inliers: int


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "RGB-D visual odometry requires ROS 2 Python packages: `rclpy`, "
            "`sensor_msgs`, `nav_msgs`, `geometry_msgs`, and `tf2_ros`."
        ) from IMPORT_ERROR
    if CV_IMPORT_ERROR is not None or cv2 is None or np is None:
        raise RuntimeError("RGB-D visual odometry requires `opencv-python`/`cv2` and `numpy`.") from CV_IMPORT_ERROR


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compose_planar(pose: PlanarPose, delta_forward_m: float, delta_yaw_rad: float) -> PlanarPose:
    mid_yaw = pose.yaw + delta_yaw_rad * 0.5
    return PlanarPose(
        x=pose.x + delta_forward_m * math.cos(mid_yaw),
        y=pose.y + delta_forward_m * math.sin(mid_yaw),
        yaw=angle_wrap(pose.yaw + delta_yaw_rad),
    )


def compose_planar_local(pose: PlanarPose, *, delta_x_m: float, delta_y_m: float, delta_yaw_rad: float) -> PlanarPose:
    mid_yaw = pose.yaw + delta_yaw_rad * 0.5
    cos_yaw = math.cos(mid_yaw)
    sin_yaw = math.sin(mid_yaw)
    world_dx = delta_x_m * cos_yaw - delta_y_m * sin_yaw
    world_dy = delta_x_m * sin_yaw + delta_y_m * cos_yaw
    return PlanarPose(
        x=pose.x + world_dx,
        y=pose.y + world_dy,
        yaw=angle_wrap(pose.yaw + delta_yaw_rad),
    )


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def imu_to_base_components(
    *,
    frame_convention: str,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    normalized = str(frame_convention).strip().lower()
    if normalized == "base_link":
        return x, y, z
    if normalized == "camera_optical":
        # ROS base_link: X forward, Y left, Z up
        # Camera optical: X right, Y down, Z forward
        return z, -x, -y
    raise ValueError(f"Unsupported IMU frame convention: {frame_convention}")


def intrinsics_from_camera_info(message: Any) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(message.k[0]),
        fy=float(message.k[4]),
        cx=float(message.k[2]),
        cy=float(message.k[5]),
    )


def image_to_array(message: Any) -> Any:
    require_runtime_dependencies()
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        array = data.reshape((height, width, 3))
        if encoding == "rgb8":
            return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if encoding in ("mono8", "8uc1"):
        return data.reshape((height, width))
    raise ValueError(f"Unsupported RGB image encoding for VO: {message.encoding}")


def depth_to_meters(message: Any) -> Any:
    require_runtime_dependencies()
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    if encoding in ("mono16", "16uc1"):
        dtype = ">u2" if bool(message.is_bigendian) else "<u2"
        raw = np.frombuffer(bytes(message.data), dtype=np.dtype(dtype)).reshape((height, width))
        return raw.astype(np.float32) / 1000.0
    if encoding == "32fc1":
        dtype = ">f4" if bool(message.is_bigendian) else "<f4"
        return np.frombuffer(bytes(message.data), dtype=np.dtype(dtype)).reshape((height, width)).astype(np.float32)
    raise ValueError(f"Unsupported depth image encoding for VO: {message.encoding}")


class FeatureRgbdOdometry:
    def __init__(self, config: RgbdVoConfig) -> None:
        require_runtime_dependencies()
        self.config = config
        self.detector = cv2.ORB_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate(self, previous: VisualOdomFrame, current: VisualOdomFrame) -> VisualOdomEstimate | None:
        prev_keypoints, prev_descriptors = self.detector.detectAndCompute(previous.gray, None)
        curr_keypoints, curr_descriptors = self.detector.detectAndCompute(current.gray, None)
        if prev_descriptors is None or curr_descriptors is None:
            return None
        matches = sorted(self.matcher.match(prev_descriptors, curr_descriptors), key=lambda item: item.distance)
        if len(matches) < self.config.min_matches:
            return None

        object_points: list[list[float]] = []
        image_points: list[list[float]] = []
        intrinsics = previous.intrinsics
        for match in matches[: min(len(matches), 200)]:
            prev_u, prev_v = prev_keypoints[match.queryIdx].pt
            curr_u, curr_v = curr_keypoints[match.trainIdx].pt
            u = int(round(prev_u))
            v = int(round(prev_v))
            if v < 0 or u < 0 or v >= previous.depth_m.shape[0] or u >= previous.depth_m.shape[1]:
                continue
            z = float(previous.depth_m[v, u])
            if not math.isfinite(z) or z < self.config.min_depth_m or z > self.config.max_depth_m:
                continue
            x = (prev_u - intrinsics.cx) * z / intrinsics.fx
            y = (prev_v - intrinsics.cy) * z / intrinsics.fy
            object_points.append([x, y, z])
            image_points.append([curr_u, curr_v])

        if len(object_points) < self.config.min_matches:
            return None

        camera_matrix = np.array(
            [
                [current.intrinsics.fx, 0.0, current.intrinsics.cx],
                [0.0, current.intrinsics.fy, current.intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            camera_matrix,
            None,
            iterationsCount=80,
            reprojectionError=5.0,
            confidence=0.99,
        )
        inlier_count = 0 if inliers is None else int(len(inliers))
        if not success or inlier_count < self.config.min_inliers:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        # solvePnP returns previous-camera coordinates expressed in the current
        # camera frame. Invert it to get current camera motion in previous frame.
        prev_from_curr = rotation.T
        prev_t_curr = -prev_from_curr @ tvec.reshape(3)
        camera_forward_m = float(prev_t_curr[2])
        camera_yaw_rad = float(math.atan2(prev_from_curr[0, 2], prev_from_curr[2, 2]))
        if abs(camera_forward_m) > self.config.max_translation_step_m:
            return None
        if abs(camera_yaw_rad) > self.config.max_yaw_step_rad:
            return None
        return VisualOdomEstimate(
            translation_m=camera_forward_m,
            yaw_rad=camera_yaw_rad,
            matches=len(matches),
            inliers=inlier_count,
        )


class RgbdVisualOdometryNode(Node):
    def __init__(self, config: RgbdVoConfig) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_rgbd_visual_odometry")
        self.config = config
        self.estimator = FeatureRgbdOdometry(config)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_publisher = self.create_publisher(Odometry, config.odom_topic, 10)
        self.latest_rgb: Any | None = None
        self.latest_depth: Any | None = None
        self.latest_imu: Any | None = None
        self.intrinsics: CameraIntrinsics | None = None
        self.previous_frame: VisualOdomFrame | None = None
        self.pose = PlanarPose(0.0, 0.0, 0.0)
        self.planar_velocity_x_m_s = 0.0
        self.planar_velocity_y_m_s = 0.0
        self._last_prediction_stamp_s: float | None = None
        self._imu_bias_started_s: float | None = None
        self._imu_bias_last_stamp_s: float | None = None
        self._imu_bias_sample_count = 0
        self._imu_bias_sum_x = 0.0
        self._imu_bias_sum_y = 0.0
        self._imu_bias_sum_z = 0.0
        self._imu_bias_ready = self.config.imu_bias_calibration_s <= 1e-6
        self._imu_bias_x_rad_s = 0.0
        self._imu_bias_y_rad_s = 0.0
        self._imu_bias_z_rad_s = 0.0
        self._latest_imu_stamp_s: float | None = None
        self._latest_imu_orientation_yaw_rad: float | None = None
        self._latest_imu_orientation_unwrapped_yaw_rad: float | None = None
        self._imu_orientation_origin_yaw_rad: float | None = None
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.create_subscription(Image, config.rgb_topic, self._on_rgb, 10)
        self.create_subscription(Image, config.depth_topic, self._on_depth, 10)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Imu, config.imu_topic, self._on_imu, 50)
        self.create_timer(1.0 / max(config.publish_rate_hz, 1e-6), self.step)
        self.get_logger().info(
            "RGB-D visual odometry ready: "
            f"rgb={config.rgb_topic} depth={config.depth_topic} camera_info={config.camera_info_topic} "
            f"imu={config.imu_topic} odom={config.odom_topic}"
        )

    def _on_rgb(self, message: Any) -> None:
        self.latest_rgb = message

    def _on_depth(self, message: Any) -> None:
        self.latest_depth = message

    def _on_camera_info(self, message: Any) -> None:
        self.intrinsics = intrinsics_from_camera_info(message)

    def _on_imu(self, message: Any) -> None:
        self.latest_imu = message
        if message.header.stamp is None:
            return
        stamp_s = stamp_to_seconds(message.header.stamp)
        self._latest_imu_stamp_s = stamp_s
        covariance = getattr(message, "orientation_covariance", None)
        if covariance is not None and len(covariance) >= 1 and float(covariance[0]) >= 0.0:
            orientation = message.orientation
            yaw_rad = yaw_from_quaternion_xyzw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            if self._latest_imu_orientation_yaw_rad is None:
                self._latest_imu_orientation_unwrapped_yaw_rad = yaw_rad
                self._imu_orientation_origin_yaw_rad = yaw_rad
            else:
                assert self._latest_imu_orientation_unwrapped_yaw_rad is not None
                self._latest_imu_orientation_unwrapped_yaw_rad += math.atan2(
                    math.sin(yaw_rad - self._latest_imu_orientation_yaw_rad),
                    math.cos(yaw_rad - self._latest_imu_orientation_yaw_rad),
                )
            self._latest_imu_orientation_yaw_rad = yaw_rad
        if self._imu_bias_ready:
            return
        if self._imu_bias_started_s is None:
            self._imu_bias_started_s = stamp_s
        if self._imu_bias_last_stamp_s is not None and stamp_s <= self._imu_bias_last_stamp_s:
            return
        self._imu_bias_last_stamp_s = stamp_s
        self._imu_bias_sum_x += float(message.angular_velocity.x)
        self._imu_bias_sum_y += float(message.angular_velocity.y)
        self._imu_bias_sum_z += float(message.angular_velocity.z)
        self._imu_bias_sample_count += 1
        elapsed_s = stamp_s - self._imu_bias_started_s
        if (
            elapsed_s >= self.config.imu_bias_calibration_s
            and self._imu_bias_sample_count >= self.config.imu_bias_min_samples
        ):
            self._imu_bias_x_rad_s = self._imu_bias_sum_x / self._imu_bias_sample_count
            self._imu_bias_y_rad_s = self._imu_bias_sum_y / self._imu_bias_sample_count
            self._imu_bias_z_rad_s = self._imu_bias_sum_z / self._imu_bias_sample_count
            self._imu_bias_ready = True
            self.get_logger().info(
                "IMU gyro bias calibrated: "
                f"x={self._imu_bias_x_rad_s:.6f} y={self._imu_bias_y_rad_s:.6f} z={self._imu_bias_z_rad_s:.6f} "
                f"from {self._imu_bias_sample_count} samples over {elapsed_s:.2f}s"
            )

    def _relative_imu_yaw_rad(self, *, stamp_s: float) -> float | None:
        if self._latest_imu_stamp_s is None:
            return None
        if abs(float(stamp_s) - self._latest_imu_stamp_s) > self.config.imu_stale_after_s:
            return None
        if self._latest_imu_orientation_unwrapped_yaw_rad is None or self._imu_orientation_origin_yaw_rad is None:
            return None
        return self._latest_imu_orientation_unwrapped_yaw_rad - self._imu_orientation_origin_yaw_rad

    def _predict_from_imu(self, *, stamp_s: float) -> tuple[float, float, float]:
        if not self._imu_bias_ready:
            self._last_prediction_stamp_s = stamp_s
            return 0.0, 0.0, 0.0
        if self.latest_imu is None or self.latest_imu.header.stamp is None:
            self._last_prediction_stamp_s = stamp_s
            return 0.0, 0.0, 0.0
        imu_stamp_s = stamp_to_seconds(self.latest_imu.header.stamp)
        if abs(stamp_s - imu_stamp_s) > self.config.imu_stale_after_s:
            self._last_prediction_stamp_s = stamp_s
            return 0.0, 0.0, 0.0
        previous_stamp_s = self._last_prediction_stamp_s
        self._last_prediction_stamp_s = stamp_s
        if previous_stamp_s is None:
            return 0.0, 0.0, 0.0
        dt = max(0.0, min(stamp_s - previous_stamp_s, 0.5))
        if dt <= 1e-6:
            return 0.0, 0.0, 0.0
        ax, ay, _ = imu_to_base_components(
            frame_convention=self.config.imu_frame_convention,
            x=float(self.latest_imu.linear_acceleration.x),
            y=float(self.latest_imu.linear_acceleration.y),
            z=float(self.latest_imu.linear_acceleration.z),
        )
        damping = max(0.0, 1.0 - self.config.imu_velocity_damping_per_s * dt)
        self.planar_velocity_x_m_s = max(
            -self.config.imu_max_planar_speed_m_s,
            min(self.config.imu_max_planar_speed_m_s, (self.planar_velocity_x_m_s + ax * dt) * damping),
        )
        self.planar_velocity_y_m_s = max(
            -self.config.imu_max_planar_speed_m_s,
            min(self.config.imu_max_planar_speed_m_s, (self.planar_velocity_y_m_s + ay * dt) * damping),
        )
        _, _, yaw_rate_rad_s = imu_to_base_components(
            frame_convention=self.config.imu_frame_convention,
            x=float(self.latest_imu.angular_velocity.x) - self._imu_bias_x_rad_s,
            y=float(self.latest_imu.angular_velocity.y) - self._imu_bias_y_rad_s,
            z=float(self.latest_imu.angular_velocity.z) - self._imu_bias_z_rad_s,
        )
        delta_yaw_rad = yaw_rate_rad_s * dt
        delta_x_m = self.planar_velocity_x_m_s * dt + 0.5 * ax * dt * dt
        delta_y_m = self.planar_velocity_y_m_s * dt + 0.5 * ay * dt * dt
        return delta_x_m, delta_y_m, delta_yaw_rad

    def step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        stamp_s = stamp_to_seconds(stamp)
        predicted_dx_m, predicted_dy_m, predicted_yaw_rad = self._predict_from_imu(stamp_s=stamp_s)
        absolute_imu_yaw_rad = self._relative_imu_yaw_rad(stamp_s=stamp_s)
        if self.latest_rgb is not None and self.latest_depth is not None and self.intrinsics is not None:
            try:
                frame = VisualOdomFrame(
                    gray=image_to_array(self.latest_rgb),
                    depth_m=depth_to_meters(self.latest_depth),
                    stamp=self.latest_rgb.header.stamp,
                    intrinsics=self.intrinsics,
                )
                stamp_s = stamp_to_seconds(frame.stamp)
                predicted_dx_m, predicted_dy_m, predicted_yaw_rad = self._predict_from_imu(stamp_s=stamp_s)
                absolute_imu_yaw_rad = self._relative_imu_yaw_rad(stamp_s=stamp_s)
                if self.previous_frame is not None:
                    estimate = self.estimator.estimate(self.previous_frame, frame)
                    if estimate is not None:
                        visual_dx_m = estimate.translation_m
                        fused_dx_m = (
                            self.config.vo_translation_weight * visual_dx_m
                            + (1.0 - self.config.vo_translation_weight) * predicted_dx_m
                        )
                        fused_dy_m = (1.0 - self.config.vo_translation_weight) * predicted_dy_m
                        if absolute_imu_yaw_rad is not None:
                            fused_yaw_rad = angle_wrap(absolute_imu_yaw_rad - self.pose.yaw)
                        elif self.latest_imu is not None:
                            fused_yaw_rad = predicted_yaw_rad
                        else:
                            fused_yaw_rad = estimate.yaw_rad
                        self.pose = compose_planar_local(
                            self.pose,
                            delta_x_m=fused_dx_m,
                            delta_y_m=fused_dy_m,
                            delta_yaw_rad=fused_yaw_rad,
                        )
                        dt = max(stamp_s - stamp_to_seconds(self.previous_frame.stamp), 1e-6)
                        self.planar_velocity_x_m_s = fused_dx_m / dt
                        self.planar_velocity_y_m_s = fused_dy_m / dt
                        self.accepted_updates += 1
                    else:
                        self.rejected_updates += 1
                self.previous_frame = frame
                stamp = frame.stamp
            except Exception as exc:
                self.rejected_updates += 1
                self.get_logger().warning(f"RGB-D VO update rejected: {exc}")
        elif (
            abs(predicted_dx_m) > 1e-6
            or abs(predicted_dy_m) > 1e-6
            or abs(predicted_yaw_rad) > 1e-6
            or absolute_imu_yaw_rad is not None
        ):
            delta_yaw_rad = (
                angle_wrap(absolute_imu_yaw_rad - self.pose.yaw)
                if absolute_imu_yaw_rad is not None
                else predicted_yaw_rad
            )
            self.pose = compose_planar_local(
                self.pose,
                delta_x_m=predicted_dx_m,
                delta_y_m=predicted_dy_m,
                delta_yaw_rad=delta_yaw_rad,
            )
        self._publish_odom(stamp)

    def _publish_odom(self, stamp: Any) -> None:
        qx, qy, qz, qw = yaw_to_quaternion_xyzw(self.pose.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.config.odom_frame
        odom.child_frame_id = self.config.base_frame
        odom.pose.pose.position.x = self.pose.x
        odom.pose.pose.position.y = self.pose.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _quaternion_msg(qx, qy, qz, qw)
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.twist.linear.x = float(self.planar_velocity_x_m_s)
        odom.twist.twist.linear.y = float(self.planar_velocity_y_m_s)
        if self.latest_imu is not None and self._imu_bias_ready:
            _, _, yaw_rate_rad_s = imu_to_base_components(
                frame_convention=self.config.imu_frame_convention,
                x=float(self.latest_imu.angular_velocity.x) - self._imu_bias_x_rad_s,
                y=float(self.latest_imu.angular_velocity.y) - self._imu_bias_y_rad_s,
                z=float(self.latest_imu.angular_velocity.z) - self._imu_bias_z_rad_s,
            )
            odom.twist.twist.angular.z = yaw_rate_rad_s
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.odom_frame
        transform.child_frame_id = self.config.base_frame
        transform.transform.translation.x = self.pose.x
        transform.transform.translation.y = self.pose.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = _quaternion_msg(qx, qy, qz, qw)
        self.tf_broadcaster.sendTransform(transform)


def _quaternion_msg(x: float, y: float, z: float, w: float) -> Any:
    msg = Quaternion()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    msg.w = float(w)
    return msg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental feature-based RGB-D visual odometry for the real XLeRobot."
    )
    parser.add_argument("--rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--depth-topic", default="/camera/head/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/head/camera_info")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="head_camera_link")
    parser.add_argument("--publish-rate-hz", type=float, default=15.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--max-translation-step-m", type=float, default=0.25)
    parser.add_argument("--max-yaw-step-deg", type=float, default=30.0)
    parser.add_argument("--imu-velocity-damping-per-s", type=float, default=0.35)
    parser.add_argument("--imu-max-planar-speed-m-s", type=float, default=0.75)
    parser.add_argument("--vo-translation-weight", type=float, default=0.85)
    parser.add_argument("--imu-stale-after-s", type=float, default=0.5)
    parser.add_argument("--imu-bias-calibration-s", type=float, default=2.0)
    parser.add_argument("--imu-bias-min-samples", type=int, default=20)
    parser.add_argument(
        "--imu-frame-convention",
        choices=("camera_optical", "base_link"),
        default="camera_optical",
        help="Interpret raw IMU vectors in camera optical coordinates or already-rotated base_link coordinates.",
    )
    parser.add_argument("--camera-x-m", type=float, default=0.0)
    parser.add_argument("--camera-y-m", type=float, default=0.0)
    parser.add_argument("--camera-yaw-rad", type=float, default=0.0)
    return parser


def config_from_args(args: argparse.Namespace) -> RgbdVoConfig:
    return RgbdVoConfig(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        imu_topic=args.imu_topic,
        odom_topic=args.odom_topic,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        camera_frame=args.camera_frame,
        publish_rate_hz=args.publish_rate_hz,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        max_translation_step_m=args.max_translation_step_m,
        max_yaw_step_rad=math.radians(args.max_yaw_step_deg),
        imu_velocity_damping_per_s=args.imu_velocity_damping_per_s,
        imu_max_planar_speed_m_s=args.imu_max_planar_speed_m_s,
        vo_translation_weight=args.vo_translation_weight,
        imu_stale_after_s=args.imu_stale_after_s,
        imu_bias_calibration_s=args.imu_bias_calibration_s,
        imu_bias_min_samples=args.imu_bias_min_samples,
        imu_frame_convention=args.imu_frame_convention,
        camera_x_m=args.camera_x_m,
        camera_y_m=args.camera_y_m,
        camera_yaw_rad=args.camera_yaw_rad,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_runtime_dependencies()
    rclpy.init()
    node = RgbdVisualOdometryNode(config_from_args(args))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
