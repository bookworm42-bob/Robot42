from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import math
import subprocess
import time
from typing import Any, Callable

import numpy as np

from xlerobot_agent.exploration import Pose2D

IMPORT_ERROR: Exception | None = None
PIL_IMPORT_ERROR: Exception | None = None
try:
    from PIL import Image as PILImage
except Exception as exc:  # pragma: no cover - optional runtime dependency.
    PIL_IMPORT_ERROR = exc
    PILImage = None

try:
    import rclpy
    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration as DurationMsg
    from geometry_msgs.msg import PoseStamped, Quaternion, Twist
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
    from rclpy.action import ActionClient
    from rclpy.action.graph import get_action_server_names_and_types_by_node
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from rclpy.time import Time as RosTime
    from sensor_msgs.msg import Image, LaserScan
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
except Exception as exc:  # pragma: no cover - runtime guard.
    IMPORT_ERROR = exc
    rclpy = None
    GoalStatus = None
    DurationMsg = None
    PoseStamped = None
    Quaternion = None
    Twist = None
    OccupancyGrid = None
    ComputePathToPose = None
    NavigateToPose = None
    Spin = None
    ActionClient = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    qos_profile_sensor_data = None
    RosTime = None
    Image = None
    LaserScan = None
    Buffer = None
    TransformListener = None
    ConnectivityException = Exception
    ExtrapolationException = Exception
    LookupException = Exception


def require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS/Nav2 exploration mode requires `rclpy`, `nav2_msgs`, `sensor_msgs`, "
            "`nav_msgs`, and `tf2_ros` in the active ROS 2 Python environment."
        ) from IMPORT_ERROR


def quaternion_from_yaw(yaw: float) -> Quaternion:
    message = Quaternion()
    message.x = 0.0
    message.y = 0.0
    message.z = math.sin(yaw / 2.0)
    message.w = math.cos(yaw / 2.0)
    return message


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def ros_goal_status_label(status: int | None) -> str:
    mapping = {
        GoalStatus.STATUS_UNKNOWN: "unknown",
        GoalStatus.STATUS_ACCEPTED: "accepted",
        GoalStatus.STATUS_EXECUTING: "executing",
        GoalStatus.STATUS_CANCELING: "canceling",
        GoalStatus.STATUS_SUCCEEDED: "succeeded",
        GoalStatus.STATUS_CANCELED: "canceled",
        GoalStatus.STATUS_ABORTED: "aborted",
    }
    return mapping.get(status, f"status_{status}")


@dataclass(frozen=True)
class RosRuntimeConfig:
    map_topic: str = "/map"
    scan_topic: str = "/scan"
    rgb_topic: str = "/camera/head/image_raw"
    cmd_vel_topic: str = "/cmd_vel"
    map_frame: str = "map"
    base_frame: str = "base_link"
    server_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    turn_scan_radians: float = math.tau
    turn_scan_timeout_s: float = 45.0
    turn_scan_settle_s: float = 1.0
    manual_spin_angular_speed_rad_s: float = 0.55
    manual_spin_publish_hz: float = 10.0
    allow_multiple_action_servers: bool = False


@dataclass(frozen=True)
class RosOccupancyMap:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    data: tuple[int, ...]

    def in_bounds(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < self.width and 0 <= cell_y < self.height

    def value(self, cell_x: int, cell_y: int) -> int:
        if not self.in_bounds(cell_x, cell_y):
            return 100
        return int(self.data[cell_y * self.width + cell_x])

    def is_unknown(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) < 0

    def is_free(self, cell_x: int, cell_y: int) -> bool:
        return self.value(cell_x, cell_y) == 0

    def is_occupied(self, cell_x: int, cell_y: int) -> bool:
        value = self.value(cell_x, cell_y)
        return value > 50 or value == 100

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((x - self.origin_x) / self.resolution)),
            int(math.floor((y - self.origin_y) / self.resolution)),
        )

    def cell_to_pose(self, cell_x: int, cell_y: int, *, yaw: float = 0.0) -> Pose2D:
        return Pose2D(
            self.origin_x + (cell_x + 0.5) * self.resolution,
            self.origin_y + (cell_y + 0.5) * self.resolution,
            yaw,
        )

    def bounds(self) -> dict[str, float]:
        return {
            "min_x": round(self.origin_x, 3),
            "max_x": round(self.origin_x + self.width * self.resolution, 3),
            "min_y": round(self.origin_y, 3),
            "max_y": round(self.origin_y + self.height * self.resolution, 3),
        }


class RosExplorationRuntime(Node):
    def __init__(self, config: RosRuntimeConfig) -> None:
        require_runtime_dependencies()
        super().__init__("xlerobot_ros_exploration_runtime")
        self.config = config
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.latest_map: RosOccupancyMap | None = None
        self.latest_map_stamp_s: float = 0.0
        self.latest_scan: LaserScan | None = None
        self.latest_image_msg: Image | None = None
        self.latest_image_data_url: str | None = None
        self._nav_goal_history: list[dict[str, Any]] = []
        self._nav_plan_history: list[dict[str, Any]] = []
        self._nav_scan_history: list[dict[str, Any]] = []
        self._cmd_vel_pub = self.create_publisher(Twist, config.cmd_vel_topic, 10)
        self._compute_path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._spin_client = ActionClient(self, Spin, "spin")

        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, config.map_topic, self._on_map, map_qos)
        self.create_subscription(LaserScan, config.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Image, config.rgb_topic, self._on_rgb, qos_profile_sensor_data)

    def _on_map(self, message: OccupancyGrid) -> None:
        self.latest_map = RosOccupancyMap(
            resolution=float(message.info.resolution),
            width=int(message.info.width),
            height=int(message.info.height),
            origin_x=float(message.info.origin.position.x),
            origin_y=float(message.info.origin.position.y),
            data=tuple(int(item) for item in message.data),
        )
        self.latest_map_stamp_s = time.time()

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan = message

    def _on_rgb(self, message: Image) -> None:
        self.latest_image_msg = message
        encoded = image_message_to_data_url(message)
        if encoded:
            self.latest_image_data_url = encoded

    def spin_until_ready(self, *, timeout_s: float | None = None) -> None:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.config.ready_timeout_s)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_map is not None and self.current_pose() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Timed out waiting for `{self.config.map_topic}` and `{self.config.map_frame}->{self.config.base_frame}` pose."
        )

    def current_pose(self) -> Pose2D | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.map_frame,
                self.config.base_frame,
                RosTime(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        )

    def spin_for(self, duration_s: float) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def compute_path(
        self,
        *,
        goal_pose: Pose2D,
        planner_id: str = "",
    ) -> tuple[int, list[Pose2D], Any]:
        if not self._compute_path_client.wait_for_server(timeout_sec=self.config.server_timeout_s):
            raise RuntimeError("`compute_path_to_pose` action server did not appear in time.")
        self._ensure_action_server_health("compute_path_to_pose")
        request = ComputePathToPose.Goal()
        request.goal = self._build_pose(goal_pose)
        if planner_id:
            request.planner_id = planner_id
        request.use_start = False
        future = self._compute_path_client.send_goal_async(request)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the ComputePathToPose goal.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        outcome = result_future.result()
        path = getattr(getattr(outcome, "result", None), "path", None)
        poses: list[Pose2D] = []
        if path is not None:
            for pose_stamped in getattr(path, "poses", []):
                pose = pose_stamped.pose
                poses.append(
                    Pose2D(
                        float(pose.position.x),
                        float(pose.position.y),
                        yaw_from_quaternion_xyzw(
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ),
                    )
                )
        return int(getattr(outcome, "status", GoalStatus.STATUS_UNKNOWN)), poses, outcome

    def navigate_to_pose(
        self,
        *,
        goal_pose: Pose2D,
        behavior_tree: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> Any:
        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=self.config.server_timeout_s):
            raise RuntimeError("`navigate_to_pose` action server did not appear in time.")
        self._ensure_action_server_health("navigate_to_pose")
        feedback_samples: list[dict[str, Any]] = []

        def _feedback(message: Any) -> None:
            feedback = getattr(message, "feedback", None)
            current_pose = self.current_pose()
            feedback_samples.append(
                {
                    "navigation_time_s": _duration_to_seconds(getattr(feedback, "navigation_time", None)),
                    "estimated_time_remaining_s": _duration_to_seconds(
                        getattr(feedback, "estimated_time_remaining", None)
                    ),
                    "distance_remaining_m": float(getattr(feedback, "distance_remaining", 0.0)),
                    "number_of_recoveries": int(getattr(feedback, "number_of_recoveries", 0)),
                    "current_pose": None if current_pose is None else current_pose.to_dict(),
                }
            )

        request = NavigateToPose.Goal()
        request.pose = self._build_pose(goal_pose)
        if behavior_tree:
            request.behavior_tree = behavior_tree
        future = self._navigate_to_pose_client.send_goal_async(request, feedback_callback=_feedback)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected the NavigateToPose goal.")
        result_future = goal_handle.get_result_async()
        cancel_requested = False
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if should_cancel is not None and should_cancel() and not cancel_requested:
                cancel_requested = True
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
        outcome = result_future.result()
        return outcome, feedback_samples

    def perform_turnaround_scan(
        self,
        *,
        reason: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        event = {
            "reason": reason,
            "mode": "nav2_spin",
            "target_yaw_rad": round(self.config.turn_scan_radians, 3),
        }
        try:
            if not self._spin_client.wait_for_server(timeout_sec=3.0):
                raise RuntimeError("Nav2 spin action server unavailable")
            self._ensure_action_server_health("spin")
            request = Spin.Goal()
            request.target_yaw = float(self.config.turn_scan_radians)
            request.time_allowance = _seconds_to_duration(self.config.turn_scan_timeout_s)
            future = self._spin_client.send_goal_async(request)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("Nav2 rejected the spin goal")
            result_future = goal_handle.get_result_async()
            cancel_requested = False
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
                if should_cancel is not None and should_cancel() and not cancel_requested:
                    cancel_requested = True
                    cancel_future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future)
            outcome = result_future.result()
            event["status"] = ros_goal_status_label(getattr(outcome, "status", GoalStatus.STATUS_UNKNOWN))
        except Exception as exc:
            event["mode"] = "manual_cmd_vel_spin"
            event["fallback_reason"] = str(exc)
            self._manual_spin(should_cancel=should_cancel)
            event["status"] = "succeeded"
        self.spin_for(self.config.turn_scan_settle_s)
        self._nav_scan_history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "module": "ros_nav2",
            "map_topic": self.config.map_topic,
            "scan_topic": self.config.scan_topic,
            "rgb_topic": self.config.rgb_topic,
            "goals": list(self._nav_goal_history),
            "plans": list(self._nav_plan_history),
            "turn_scans": list(self._nav_scan_history),
        }

    def record_goal(self, payload: dict[str, Any]) -> None:
        self._nav_goal_history.append(payload)

    def record_plan(self, payload: dict[str, Any]) -> None:
        self._nav_plan_history.append(payload)

    def close(self) -> None:
        self.destroy_node()

    def _manual_spin(self, *, should_cancel: Callable[[], bool] | None = None) -> None:
        twist = Twist()
        twist.angular.z = float(self.config.manual_spin_angular_speed_rad_s)
        duration_s = abs(self.config.turn_scan_radians) / max(abs(twist.angular.z), 1e-6)
        step_s = 1.0 / max(self.config.manual_spin_publish_hz, 1e-6)
        deadline = time.time() + duration_s
        while time.time() < deadline:
            if should_cancel is not None and should_cancel():
                break
            self._cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(step_s)
        self._cmd_vel_pub.publish(Twist())

    def _action_servers(self, action_name: str) -> list[str]:
        normalized_action = action_name if action_name.startswith("/") else f"/{action_name}"
        servers: set[str] = set()
        for node_name, namespace in self.get_node_names_and_namespaces():
            try:
                action_servers = get_action_server_names_and_types_by_node(self, node_name, namespace)
            except Exception:
                continue
            for advertised_name, _types in action_servers:
                normalized_advertised = (
                    advertised_name if advertised_name.startswith("/") else f"/{advertised_name}"
                )
                if normalized_advertised != normalized_action:
                    continue
                if not namespace or namespace == "/":
                    servers.add(f"/{node_name}")
                else:
                    servers.add(f"{namespace.rstrip('/')}/{node_name}")
                break
        return sorted(servers)

    def _action_servers_via_cli(self, action_name: str) -> list[str] | None:
        normalized_action = action_name if action_name.startswith("/") else f"/{action_name}"
        try:
            completed = subprocess.run(
                ["ros2", "action", "info", normalized_action],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        servers: list[str] = []
        in_servers = False
        for raw_line in completed.stdout.splitlines():
            line = raw_line.rstrip()
            if line.startswith("Action servers:"):
                in_servers = True
                continue
            if line.startswith("Action clients:"):
                in_servers = False
                continue
            if in_servers and line.strip():
                servers.append(line.strip())
        return servers

    def _ensure_action_server_health(self, action_name: str) -> None:
        if self.config.allow_multiple_action_servers:
            return
        servers: list[str] = []
        for _attempt in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
            servers = self._action_servers(action_name)
            if len(servers) > 1:
                break
            time.sleep(0.05)
        if len(servers) <= 1:
            servers = self._action_servers_via_cli(action_name) or servers
        if len(servers) <= 1:
            return
        raise RuntimeError(
            f"Expected exactly one action server for `{action_name}`, found {len(servers)}: {', '.join(servers)}."
        )

    def _build_pose(self, pose: Pose2D) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.frame_id = self.config.map_frame
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.pose.position.x = float(pose.x)
        stamped.pose.position.y = float(pose.y)
        stamped.pose.position.z = 0.0
        stamped.pose.orientation = quaternion_from_yaw(float(pose.yaw))
        return stamped


def image_message_to_data_url(message: Image) -> str | None:
    if PILImage is None:
        return None
    if str(message.encoding).lower() not in {"rgb8", "bgr8"}:
        return None
    channels = 3
    expected_bytes = int(message.height) * int(message.width) * channels
    if len(message.data) < expected_bytes:
        return None
    array = np.frombuffer(message.data, dtype=np.uint8, count=expected_bytes).reshape(
        int(message.height),
        int(message.width),
        channels,
    )
    if str(message.encoding).lower() == "bgr8":
        array = array[..., ::-1]
    image = PILImage.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def path_length_m(poses: list[Pose2D]) -> float:
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for previous, nxt in zip(poses, poses[1:]):
        total += math.dist((previous.x, previous.y), (nxt.x, nxt.y))
    return total


def seconds_since(stamp_s: float) -> float:
    if stamp_s <= 0:
        return 1e9
    return max(time.time() - stamp_s, 0.0)


def _seconds_to_duration(value_s: float) -> DurationMsg:
    value_s = max(float(value_s), 0.0)
    seconds = int(value_s)
    nanoseconds = int((value_s - seconds) * 1e9)
    return DurationMsg(sec=seconds, nanosec=nanoseconds)


def _duration_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    sec = getattr(value, "sec", None)
    nanosec = getattr(value, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) / 1e9
