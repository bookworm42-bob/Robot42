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
    from sensor_msgs.msg import CameraInfo, Image
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


def compose_planar(pose: PlanarPose, delta_forward_m: float, delta_yaw_rad: float) -> PlanarPose:
    mid_yaw = pose.yaw + delta_yaw_rad * 0.5
    return PlanarPose(
        x=pose.x + delta_forward_m * math.cos(mid_yaw),
        y=pose.y + delta_forward_m * math.sin(mid_yaw),
        yaw=angle_wrap(pose.yaw + delta_yaw_rad),
    )


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
        self.intrinsics: CameraIntrinsics | None = None
        self.previous_frame: VisualOdomFrame | None = None
        self.pose = PlanarPose(0.0, 0.0, 0.0)
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.create_subscription(Image, config.rgb_topic, self._on_rgb, 10)
        self.create_subscription(Image, config.depth_topic, self._on_depth, 10)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, 10)
        self.create_timer(1.0 / max(config.publish_rate_hz, 1e-6), self.step)
        self.get_logger().info(
            "RGB-D visual odometry ready: "
            f"rgb={config.rgb_topic} depth={config.depth_topic} camera_info={config.camera_info_topic} "
            f"odom={config.odom_topic}"
        )

    def _on_rgb(self, message: Any) -> None:
        self.latest_rgb = message

    def _on_depth(self, message: Any) -> None:
        self.latest_depth = message

    def _on_camera_info(self, message: Any) -> None:
        self.intrinsics = intrinsics_from_camera_info(message)

    def step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        if self.latest_rgb is not None and self.latest_depth is not None and self.intrinsics is not None:
            try:
                frame = VisualOdomFrame(
                    gray=image_to_array(self.latest_rgb),
                    depth_m=depth_to_meters(self.latest_depth),
                    stamp=self.latest_rgb.header.stamp,
                    intrinsics=self.intrinsics,
                )
                if self.previous_frame is not None:
                    estimate = self.estimator.estimate(self.previous_frame, frame)
                    if estimate is not None:
                        base_forward_m = estimate.translation_m
                        base_yaw_rad = estimate.yaw_rad
                        self.pose = compose_planar(self.pose, base_forward_m, base_yaw_rad)
                        self.accepted_updates += 1
                    else:
                        self.rejected_updates += 1
                self.previous_frame = frame
                stamp = frame.stamp
            except Exception as exc:
                self.rejected_updates += 1
                self.get_logger().warning(f"RGB-D VO update rejected: {exc}")
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
    parser.add_argument("--camera-x-m", type=float, default=0.0)
    parser.add_argument("--camera-y-m", type=float, default=0.0)
    parser.add_argument("--camera-yaw-rad", type=float, default=0.0)
    return parser


def config_from_args(args: argparse.Namespace) -> RgbdVoConfig:
    return RgbdVoConfig(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
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
