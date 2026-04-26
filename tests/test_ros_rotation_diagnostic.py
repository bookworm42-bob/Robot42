from __future__ import annotations

import math
import unittest

from xlerobot_playground.ros_rotation_diagnostic import (
    compute_control_angular_velocity,
    feedback_unwrapped_yaw_rad,
)


class RosRotationDiagnosticTests(unittest.TestCase):
    def test_feedback_uses_imu_orientation_when_available(self) -> None:
        sample = {
            "imu_orientation_available": True,
            "imu_orientation_unwrapped_yaw_rad": 1.25,
            "imu_unwrapped_yaw_rad": 0.8,
        }

        actual = feedback_unwrapped_yaw_rad(sample, source="imu")

        self.assertAlmostEqual(actual, 1.25)

    def test_feedback_falls_back_to_requested_source_key(self) -> None:
        sample = {"tf_unwrapped_yaw_rad": 0.75}

        actual = feedback_unwrapped_yaw_rad(sample, source="tf")

        self.assertAlmostEqual(actual, 0.75)

    def test_control_runs_open_loop_without_feedback(self) -> None:
        command, reached = compute_control_angular_velocity(
            requested_angular_rad_s=0.3,
            target_yaw_rad=math.radians(90.0),
            feedback_yaw_rad=None,
        )

        self.assertAlmostEqual(command, 0.3)
        self.assertFalse(reached)

    def test_control_slows_down_near_target(self) -> None:
        command, reached = compute_control_angular_velocity(
            requested_angular_rad_s=0.3,
            target_yaw_rad=math.radians(90.0),
            feedback_yaw_rad=math.radians(70.0),
        )

        self.assertGreater(command, 0.0)
        self.assertLess(command, 0.3)
        self.assertFalse(reached)

    def test_control_stops_within_tolerance(self) -> None:
        command, reached = compute_control_angular_velocity(
            requested_angular_rad_s=0.3,
            target_yaw_rad=math.radians(90.0),
            feedback_yaw_rad=math.radians(89.2),
        )

        self.assertEqual(command, 0.0)
        self.assertTrue(reached)


if __name__ == "__main__":
    unittest.main()
