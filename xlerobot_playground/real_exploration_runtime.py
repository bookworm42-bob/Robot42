from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import resolve_xlerobot_repo_root
from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import (
    NavigationDependency,
    NavigationResult,
    RGBDObservation,
    ScanObservation,
)


@dataclass(frozen=True)
class RealXLeRobotRuntimeConfig:
    repo_root: str = str(resolve_xlerobot_repo_root())
    robot_kind: str = "xlerobot_2wheels"
    port1: str = "/dev/tty.usbmodem5B140330101"
    port2: str = "/dev/tty.usbmodem5B140332271"
    fps: int = 30
    use_degrees: bool = False
    camera_frame: str = "head_camera_link"
    allow_motion_commands: bool = False
    max_linear_m_s: float = 0.20
    max_angular_rad_s: float = 0.50


class RealXLeRobotDirectRuntime:
    """Direct real-XLeRobot exploration embodiment.

    This class is intentionally a safe skeleton. It names the same contract as
    the ManiSkill runtime, while leaving Orbbec RGB-D, pose tracking, and base
    command conversion unimplemented until those pieces can be tested on
    hardware with a deadman stop.
    """

    def __init__(
        self,
        config: RealXLeRobotRuntimeConfig | None = None,
        *,
        navigation: NavigationDependency | None = None,
        robot: Any | None = None,
    ) -> None:
        self.config = config or RealXLeRobotRuntimeConfig()
        self.navigation = navigation
        self._robot = robot
        self._connected = False

    @property
    def name(self) -> str:
        return "real_xlerobot_direct"

    @property
    def robot(self) -> Any:
        if self._robot is None:
            interface = XLeRobotInterface(self.config.repo_root)
            if self.config.robot_kind == "xlerobot_2wheels":
                config_cls, robot_cls = interface.robot_2wheels_classes()
            else:
                config_cls, robot_cls = interface.robot_classes()
            self._robot = robot_cls(
                config_cls(
                    port1=self.config.port1,
                    port2=self.config.port2,
                    cameras={},
                    use_degrees=self.config.use_degrees,
                )
            )
        return self._robot

    def connect(self) -> None:
        if self._connected:
            return
        self.robot.connect()
        self._connected = True

    def close(self) -> None:
        if not self._connected:
            return
        self.robot.disconnect()
        self._connected = False

    def reset(self) -> None:
        raise NotImplementedError("Real reset/home behavior is not implemented yet.")

    def current_pose(self) -> Pose2D:
        raise NotImplementedError("Real pose tracking requires odometry or localization integration.")

    def capture_rgbd(self) -> RGBDObservation:
        raise NotImplementedError("Real RGB-D capture requires Orbbec depth integration.")

    def latest_scan(self) -> ScanObservation | None:
        raise NotImplementedError("Real scan requires Orbbec depth-to-scan integration.")

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> NavigationResult:
        if not self.config.allow_motion_commands:
            return NavigationResult(
                succeeded=False,
                message="Real motion commands are disabled. Set allow_motion_commands=True after hardware checks.",
                metadata={
                    "requested_linear_m_s": linear_m_s,
                    "requested_angular_rad_s": angular_rad_s,
                },
            )
        self.connect()
        action = self._base_velocity_action(linear_m_s=linear_m_s, angular_rad_s=angular_rad_s)
        sent = self.robot.send_action(action)
        return NavigationResult(
            succeeded=True,
            message="Velocity command sent to real XLeRobot base.",
            metadata={"requested_action": action, "sent_action": sent},
        )

    def stop(self) -> NavigationResult:
        if not self._connected:
            return NavigationResult(succeeded=True, message="Robot is not connected; nothing to stop.")
        if hasattr(self.robot, "stop_base"):
            self.robot.stop_base()
            return NavigationResult(succeeded=True, message="Real XLeRobot base stop command sent.")
        sent = self.robot.send_action(self._base_velocity_action(linear_m_s=0.0, angular_rad_s=0.0))
        return NavigationResult(
            succeeded=True,
            message="Zero velocity command sent to real XLeRobot base.",
            metadata={"sent_action": sent},
        )

    def rotate_in_place(self, yaw_delta_rad: float) -> NavigationResult:
        raise NotImplementedError("Real rotate-in-place requires the base velocity adapter and odometry feedback.")

    def execute_path(self, path: Sequence[Pose2D]) -> NavigationResult:
        raise NotImplementedError("Real path execution requires navigation dependency wiring and command safety.")

    def _base_velocity_action(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, float]:
        linear = max(-self.config.max_linear_m_s, min(self.config.max_linear_m_s, float(linear_m_s)))
        angular = max(-self.config.max_angular_rad_s, min(self.config.max_angular_rad_s, float(angular_rad_s)))
        action = {
            "x.vel": linear,
            "theta.vel": math.degrees(angular),
        }
        if self.config.robot_kind != "xlerobot_2wheels":
            action["y.vel"] = 0.0
        return action
