from __future__ import annotations

import unittest

from xlerobot_playground.ros_forward_accel_diagnostic import (
    forward_displacement_m,
    gravity_components_from_tilt,
    integrate_acceleration_step,
    summarize,
    tilt_correction_alpha_for_motion,
    tilt_from_acceleration,
    update_tilt_estimate,
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

    def test_tilt_from_acceleration_detects_level_pose(self) -> None:
        roll_rad, pitch_rad = tilt_from_acceleration(
            accel_x_m_s2=0.0,
            accel_y_m_s2=0.0,
            accel_z_m_s2=9.81,
        )

        self.assertAlmostEqual(roll_rad, 0.0)
        self.assertAlmostEqual(pitch_rad, 0.0)

    def test_gravity_components_from_tilt_projects_forward_gravity(self) -> None:
        gravity_forward, gravity_lateral, gravity_vertical = gravity_components_from_tilt(
            roll_rad=0.0,
            pitch_rad=0.1,
            gravity_m_s2=9.81,
        )

        self.assertAlmostEqual(gravity_forward, 9.81 * 0.09983341664682815, places=6)
        self.assertAlmostEqual(gravity_lateral, 0.0, places=6)
        self.assertAlmostEqual(gravity_vertical, 9.81 * 0.9950041652780258, places=6)

    def test_update_tilt_estimate_blends_gyro_and_accel(self) -> None:
        roll_rad, pitch_rad = update_tilt_estimate(
            previous_roll_rad=0.0,
            previous_pitch_rad=0.0,
            gyro_roll_rate_rad_s=0.2,
            gyro_pitch_rate_rad_s=0.0,
            accel_x_m_s2=0.0,
            accel_y_m_s2=0.0,
            accel_z_m_s2=9.81,
            dt_s=0.1,
            accel_correction_alpha=0.25,
        )

        self.assertAlmostEqual(roll_rad, 0.015)
        self.assertAlmostEqual(pitch_rad, 0.0)

    def test_tilt_correction_alpha_for_motion_disables_accel_correction_while_moving(self) -> None:
        self.assertEqual(
            tilt_correction_alpha_for_motion(
                stationary_alpha=0.02,
                commanded_linear_m_s=0.03,
                moving_alpha=0.0,
            ),
            0.0,
        )
        self.assertEqual(
            tilt_correction_alpha_for_motion(
                stationary_alpha=0.02,
                commanded_linear_m_s=0.0,
                moving_alpha=0.0,
            ),
            0.02,
        )

    def test_summary_reports_accel_and_pose_distance(self) -> None:
        summary = summarize(
            [
                {
                    "t_s": 0.0,
                    "imu_available": True,
                    "imu_timestamp_s": 10.0,
                    "imu_estimated_forward_distance_m": 0.0,
                    "imu_estimated_forward_velocity_m_s": 0.0,
                    "imu_used_forward_acceleration_m_s2": 0.0,
                    "imu_stationary": True,
                    "imu_zupt_applied": True,
                    "tf_forward_distance_m": 0.0,
                    "tf_lateral_distance_m": 0.0,
                    "odom_forward_distance_m": 0.0,
                    "odom_lateral_distance_m": 0.0,
                    "accel_bias_forward_m_s2": 0.01,
                    "accel_bias_lateral_m_s2": 0.02,
                    "accel_bias_vertical_m_s2": 9.81,
                    "gyro_bias_roll_rad_s": 0.001,
                    "gyro_bias_pitch_rad_s": 0.002,
                    "gyro_bias_yaw_rad_s": 0.003,
                    "imu_gravity_forward_m_s2": 0.1,
                    "imu_tilt_accel_correction_alpha": 0.0,
                },
                {
                    "t_s": 3.0,
                    "imu_available": True,
                    "imu_timestamp_s": 13.0,
                    "imu_estimated_forward_distance_m": 0.44,
                    "imu_estimated_forward_velocity_m_s": 0.02,
                    "imu_used_forward_acceleration_m_s2": -0.12,
                    "imu_stationary": True,
                    "imu_zupt_applied": False,
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
        self.assertEqual(summary["accelerometer"]["stationary_sample_count"], 2)
        self.assertEqual(summary["accelerometer"]["zupt_applied_sample_count"], 1)
        self.assertAlmostEqual(summary["accelerometer"]["observed_imu_rate_hz"], 0.333)
        self.assertAlmostEqual(summary["tf"]["forward_distance_m"], 0.43)
        self.assertAlmostEqual(summary["odom_topic"]["forward_distance_m"], 0.42)
        self.assertTrue(summary["accelerometer_bias_applied"])
        self.assertTrue(summary["tilt_compensation_applied"])
        self.assertAlmostEqual(summary["accelerometer_bias"]["gyro_roll_rad_s"], 0.001)


if __name__ == "__main__":
    unittest.main()
