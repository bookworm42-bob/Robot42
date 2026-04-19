from __future__ import annotations

import argparse
import math
import unittest

from xlerobot_playground.robot_brain_smoke_test import (
    Pose2D,
    SmokeStep,
    relative_goal,
    steps_from_args,
    velocity_command,
)


class RobotBrainSmokeTestTests(unittest.TestCase):
    def test_relative_forward_goal_uses_pose_heading(self) -> None:
        goal = relative_goal(Pose2D(1.0, 2.0, math.pi / 2.0), SmokeStep("forward", translation_m=0.05))

        self.assertAlmostEqual(goal.x, 1.0)
        self.assertAlmostEqual(goal.y, 2.05)
        self.assertAlmostEqual(goal.yaw, math.pi / 2.0)

    def test_steps_are_mode_b_small_forward_and_rotations(self) -> None:
        args = argparse.Namespace(forward_m=0.05, turn_deg=5.0)

        steps = steps_from_args(args)

        self.assertEqual([step.name for step in steps], ["forward", "rotate_left", "rotate_right"])
        self.assertEqual(steps[0].translation_m, 0.05)
        self.assertAlmostEqual(steps[1].yaw_delta_rad, math.radians(5.0))
        self.assertAlmostEqual(steps[2].yaw_delta_rad, math.radians(-5.0))

    def test_velocity_command_rotates_before_driving(self) -> None:
        linear, angular, done = velocity_command(
            Pose2D(0.0, 0.0, math.pi / 2.0),
            Pose2D(1.0, 0.0, 0.0),
            max_linear_m_s=0.03,
            max_angular_rad_s=0.10,
            yaw_align_tolerance_rad=0.1,
            goal_distance_tolerance_m=0.02,
        )

        self.assertEqual(linear, 0.0)
        self.assertLess(angular, 0.0)
        self.assertFalse(done)

    def test_velocity_command_finishes_inside_pose_tolerances(self) -> None:
        linear, angular, done = velocity_command(
            Pose2D(0.0, 0.0, 0.01),
            Pose2D(0.01, 0.0, 0.0),
            max_linear_m_s=0.03,
            max_angular_rad_s=0.10,
            yaw_align_tolerance_rad=0.1,
            goal_distance_tolerance_m=0.02,
        )

        self.assertEqual((linear, angular), (0.0, 0.0))
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
