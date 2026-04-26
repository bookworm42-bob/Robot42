from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.ros_nav2_runtime import (
    RosExplorationRuntime,
    compute_turn_command,
    remaining_turn_delta_rad,
)


class RosNav2RuntimeTests(unittest.TestCase):
    def test_compute_turn_command_stops_at_target(self) -> None:
        command, done = compute_turn_command(
            requested_angular_rad_s=0.3,
            target_yaw_rad=math.radians(90.0),
            feedback_yaw_rad=math.radians(89.5),
        )

        self.assertEqual(command, 0.0)
        self.assertTrue(done)

    def test_remaining_turn_delta_catches_up_segment_shortfall(self) -> None:
        self.assertAlmostEqual(
            remaining_turn_delta_rad(
                desired_total_yaw_rad=math.radians(60.0),
                achieved_total_yaw_rad=math.radians(28.0),
            ),
            math.radians(32.0),
        )

    def test_remaining_turn_delta_clamps_when_total_target_already_met(self) -> None:
        self.assertEqual(
            remaining_turn_delta_rad(
                desired_total_yaw_rad=math.radians(30.0),
                achieved_total_yaw_rad=math.radians(31.0),
            ),
            0.0,
        )

    def test_camera_pan_scan_captures_outward_sweeps_only_and_restores_center(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    robot_brain_url="http://brain.local:8765",
                    camera_pan_action_key="head_motor_1.pos",
                    camera_pan_settle_s=0.0,
                    turn_scan_settle_s=0.0,
                    server_timeout_s=1.0,
                )
                self.commands: list[float] = []
                self.hold_count = 0
                self.scan_observations = [{"pose": Pose2D(0.0, 0.0, 99.0)} for _ in range(99)]
                self._nav_scan_history = []

            def current_pose(self) -> Pose2D:
                return Pose2D(1.0, 2.0, 0.25)

            def _command_camera_pan(self, pan_rad: float, **_kwargs):
                self.commands.append(pan_rad)
                return {"pan_rad": pan_rad}

            def _capture_settled_scan_observation(self):
                return {
                    "pose": Pose2D(1.0, 2.0, self.commands[-1]),
                    "ranges": (1.0,),
                    "angle_min": 0.0,
                    "angle_increment": 0.1,
                    "range_min": 0.05,
                    "range_max": 4.0,
                }

            def hold_stop_until_stable(self, *, duration_s: float):
                self.hold_count += 1
                return {"stable": True, "duration_s": duration_s}

            def drain_scan_observations(self, since_index: int):
                return list(self.scan_observations[since_index:]), len(self.scan_observations)

        runtime = FakeRuntime()

        result = RosExplorationRuntime._perform_camera_pan_scan(
            runtime,
            reason="test",
            should_cancel=None,
            start_time=0.0,
            start_pose=Pose2D(1.0, 2.0, 0.25),
            observation_start_index=10,
            sample_count=6,
            event={"reason": "test", "mode": "camera_pan", "sample_count": 6},
        )

        expected = [0.0, -math.pi / 2.0, -math.pi, 0.0, 0.0, math.pi / 2.0, math.pi, 0.0, 0.0]
        self.assertEqual(len(runtime.commands), len(expected))
        for actual, expected_value in zip(runtime.commands, expected):
            self.assertAlmostEqual(actual, expected_value)
        self.assertEqual(len(result["observations"]), 6)
        self.assertEqual([item["scan_sweep"] for item in result["observations"]], ["left"] * 3 + ["right"] * 3)
        self.assertEqual(result["observation_stop_index"], 99)
        self.assertEqual(result["raw_observation_count"], 89)
        self.assertEqual(result["scan_stop_reason"], "completed")

    def test_camera_pan_scan_pose_keeps_tf_head_yaw(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(publish_internal_navigation_map=True)
                self._use_turn_feedback_for_scan_pose = False
                self._scan_sensor_yaw_offset_rad = None

            def _current_turn_feedback(self):
                return "imu", math.radians(10.0)

        sensor_pose = Pose2D(1.0, 2.0, math.radians(75.0))

        adjusted = RosExplorationRuntime._scan_pose_with_turn_feedback(FakeRuntime(), sensor_pose)

        self.assertEqual(adjusted, sensor_pose)

    def test_robot_spin_scan_pose_can_use_imu_yaw_feedback(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.config = SimpleNamespace(publish_internal_navigation_map=True)
                self._use_turn_feedback_for_scan_pose = True
                self._scan_sensor_yaw_offset_rad = None
                self.feedback_yaw = math.radians(10.0)

            def _current_turn_feedback(self):
                return "imu", self.feedback_yaw

        runtime = FakeRuntime()

        first = RosExplorationRuntime._scan_pose_with_turn_feedback(
            runtime,
            Pose2D(1.0, 2.0, math.radians(75.0)),
        )
        runtime.feedback_yaw = math.radians(30.0)
        second = RosExplorationRuntime._scan_pose_with_turn_feedback(
            runtime,
            Pose2D(1.0, 2.0, math.radians(75.0)),
        )

        self.assertAlmostEqual(first.yaw, math.radians(75.0))
        self.assertAlmostEqual(second.yaw, math.radians(95.0))

    def test_unknown_turn_scan_mode_is_rejected(self) -> None:
        class FakeRuntime:
            config = SimpleNamespace(
                turn_scan_mode="bad",
                camera_pan_sample_count=12,
                turn_scan_radians=math.tau,
            )
            scan_observations = []

            def current_pose(self):
                return None

        with self.assertRaises(ValueError):
            RosExplorationRuntime.perform_turnaround_scan(FakeRuntime(), reason="test")


if __name__ == "__main__":
    unittest.main()
