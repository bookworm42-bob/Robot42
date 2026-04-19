from __future__ import annotations

import math
import unittest

from xlerobot_playground.real_nav2_smoke_test import (
    OdomPose2D,
    SmokeStep,
    angle_wrap,
    distance_error_m,
    relative_goal,
    steps_from_args,
    yaw_error_rad,
)


class _Args:
    forward_m = 0.10
    turn_deg = 10.0


class RealNav2SmokeTestMathTests(unittest.TestCase):
    def test_relative_forward_goal_uses_current_yaw(self) -> None:
        start = OdomPose2D(1.0, 2.0, math.pi / 2.0)

        goal = relative_goal(start, SmokeStep("forward", translation_m=0.10))

        self.assertAlmostEqual(goal.x, 1.0)
        self.assertAlmostEqual(goal.y, 2.10)
        self.assertAlmostEqual(goal.yaw, math.pi / 2.0)

    def test_relative_rotation_goal_keeps_position(self) -> None:
        start = OdomPose2D(1.0, 2.0, 0.0)

        goal = relative_goal(start, SmokeStep("left", yaw_delta_rad=math.radians(10.0)))

        self.assertEqual(goal.x, 1.0)
        self.assertEqual(goal.y, 2.0)
        self.assertAlmostEqual(goal.yaw, math.radians(10.0))

    def test_angle_wrap_crosses_pi_boundary(self) -> None:
        wrapped = angle_wrap(math.radians(190.0))

        self.assertAlmostEqual(wrapped, math.radians(-170.0))

    def test_pose_errors(self) -> None:
        actual = OdomPose2D(0.03, 0.04, math.radians(355.0))
        goal = OdomPose2D(0.0, 0.0, math.radians(5.0))

        self.assertAlmostEqual(distance_error_m(actual, goal), 0.05)
        self.assertAlmostEqual(yaw_error_rad(actual, goal), math.radians(10.0))

    def test_default_steps_from_args_are_small_and_symmetric(self) -> None:
        steps = steps_from_args(_Args())

        self.assertEqual([step.name for step in steps], ["forward", "rotate_left", "rotate_right"])
        self.assertEqual(steps[0].translation_m, 0.10)
        self.assertAlmostEqual(steps[1].yaw_delta_rad, math.radians(10.0))
        self.assertAlmostEqual(steps[2].yaw_delta_rad, math.radians(-10.0))


if __name__ == "__main__":
    unittest.main()
