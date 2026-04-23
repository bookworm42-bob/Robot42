from __future__ import annotations

import unittest

from xlerobot_playground.ros_forward_accel_diagnostic import (
    forward_displacement_m,
    integrate_acceleration_step,
    summarize,
)


class RosForwardAccelDiagnosticTests(unittest.TestCase):
    def test_integrate_acceleration_step_advances_distance_and_velocity(self) -> None:
        distance_m, velocity_m_s, used_accel = integrate_acceleration_step(
            distance_m=0.0,
            velocity_m_s=0.0,
            acceleration_m_s2=1.0,
            dt_s=0.5,
            acceleration_deadband_m_s2=0.0,
            velocity_damping_per_s=0.0,
            max_velocity_m_s=10.0,
        )

        self.assertAlmostEqual(velocity_m_s, 0.5)
        self.assertAlmostEqual(distance_m, 0.125)
        self.assertEqual(used_accel, 1.0)

    def test_integrate_acceleration_step_deadbands_small_noise(self) -> None:
        distance_m, velocity_m_s, used_accel = integrate_acceleration_step(
            distance_m=0.2,
            velocity_m_s=0.1,
            acceleration_m_s2=0.03,
            dt_s=0.2,
            acceleration_deadband_m_s2=0.08,
            velocity_damping_per_s=0.0,
            max_velocity_m_s=10.0,
        )

        self.assertEqual(used_accel, 0.0)
        self.assertAlmostEqual(velocity_m_s, 0.1)
        self.assertAlmostEqual(distance_m, 0.22)

    def test_forward_displacement_projects_into_start_heading(self) -> None:
        forward_m, lateral_m = forward_displacement_m(
            start_x_m=1.0,
            start_y_m=2.0,
            start_yaw_rad=0.0,
            current_x_m=1.45,
            current_y_m=2.05,
        )

        self.assertAlmostEqual(forward_m, 0.45)
        self.assertAlmostEqual(lateral_m, 0.05)

    def test_summary_reports_accel_and_pose_distance(self) -> None:
        summary = summarize(
            [
                {
                    "t_s": 0.0,
                    "imu_available": True,
                    "imu_estimated_forward_distance_m": 0.0,
                    "imu_estimated_forward_velocity_m_s": 0.0,
                    "imu_used_forward_acceleration_m_s2": 0.0,
                    "tf_forward_distance_m": 0.0,
                    "tf_lateral_distance_m": 0.0,
                    "odom_forward_distance_m": 0.0,
                    "odom_lateral_distance_m": 0.0,
                    "accel_bias_forward_m_s2": 0.01,
                    "accel_bias_lateral_m_s2": 0.02,
                    "accel_bias_vertical_m_s2": 9.81,
                },
                {
                    "t_s": 3.0,
                    "imu_available": True,
                    "imu_estimated_forward_distance_m": 0.44,
                    "imu_estimated_forward_velocity_m_s": 0.02,
                    "imu_used_forward_acceleration_m_s2": -0.12,
                    "tf_forward_distance_m": 0.43,
                    "tf_lateral_distance_m": 0.01,
                    "odom_forward_distance_m": 0.42,
                    "odom_lateral_distance_m": 0.02,
                    "stop_reason": "target_accel_distance_reached",
                },
            ]
        )

        self.assertEqual(summary["stop_reason"], "target_accel_distance_reached")
        self.assertEqual(summary["accelerometer"]["valid_sample_count"], 2)
        self.assertAlmostEqual(summary["accelerometer"]["reported_distance_m"], 0.44)
        self.assertAlmostEqual(summary["tf"]["forward_distance_m"], 0.43)
        self.assertAlmostEqual(summary["odom_topic"]["forward_distance_m"], 0.42)


if __name__ == "__main__":
    unittest.main()
