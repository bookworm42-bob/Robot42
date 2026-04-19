from __future__ import annotations

import argparse
import math
import unittest

from xlerobot_playground.robot_brain_smoke_test import (
    FollowResult,
    Pose2D,
    RobotBrainClient,
    SmokeStep,
    build_parser,
    execute_velocity_for_duration,
    follow_goal,
    forward_translation_reached,
    relative_goal,
    run_smoke_test,
    steps_from_args,
    velocity_command,
)


class FakeRobotClient:
    def __init__(self) -> None:
        self.commands = []

    def cmd_vel(self, *, linear_m_s: float, angular_rad_s: float):
        self.commands.append((linear_m_s, angular_rad_s))
        return {"succeeded": True}

    def zero_velocity(self):
        return self.cmd_vel(linear_m_s=0.0, angular_rad_s=0.0)


class StaticRouter:
    def __init__(self, pose: Pose2D) -> None:
        self.pose = pose

    def current_pose(self) -> Pose2D:
        return self.pose


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

    def test_parser_accepts_separate_robot_timeout(self) -> None:
        args = build_parser().parse_args(
            [
                "--router-url",
                "http://router:8891",
                "--robot-timeout-s",
                "12",
                "--debug-progress",
            ]
        )

        self.assertEqual(args.robot_timeout_s, 12.0)
        self.assertTrue(args.debug_progress)
        self.assertEqual(args.pose_validation_mode, "diagnostic")

    def test_parser_accepts_strict_pose_validation(self) -> None:
        args = build_parser().parse_args(
            [
                "--router-url",
                "http://router:8891",
                "--pose-validation-mode",
                "strict",
            ]
        )

        self.assertEqual(args.pose_validation_mode, "strict")

    def test_parser_allows_motor_smoke_without_router(self) -> None:
        args = build_parser().parse_args(["--motor-smoke-only"])

        self.assertTrue(args.motor_smoke_only)
        self.assertIsNone(args.router_url)

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

    def test_forward_translation_reached_accepts_forward_overshoot(self) -> None:
        self.assertTrue(
            forward_translation_reached(
                start=Pose2D(0.0, 0.0, 0.0),
                goal=Pose2D(0.05, 0.0, 0.0),
                pose=Pose2D(0.08, 0.01, 0.02),
                lateral_tolerance_m=0.025,
                yaw_tolerance_rad=0.1,
            )
        )

    def test_forward_translation_reached_rejects_lateral_drift(self) -> None:
        self.assertFalse(
            forward_translation_reached(
                start=Pose2D(0.0, 0.0, 0.0),
                goal=Pose2D(0.05, 0.0, 0.0),
                pose=Pose2D(0.08, 0.04, 0.0),
                lateral_tolerance_m=0.025,
                yaw_tolerance_rad=0.1,
            )
        )

    def test_execute_velocity_for_duration_sends_zero_cleanup(self) -> None:
        robot = FakeRobotClient()

        result = execute_velocity_for_duration(
            robot=robot,
            linear_m_s=0.01,
            angular_rad_s=0.0,
            duration_s=0.0,
            publish_hz=10.0,
        )

        self.assertEqual(result["command_count"], 0)
        self.assertEqual(robot.commands, [(0.0, 0.0)])

    def test_follow_goal_reports_timeout_without_raising(self) -> None:
        robot = FakeRobotClient()

        result = follow_goal(
            router=StaticRouter(Pose2D(0.0, 0.0, 0.0)),
            robot=robot,
            start=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(1.0, 0.0, 0.0),
            step=SmokeStep("forward", translation_m=1.0),
            timeout_s=0.0,
            publish_hz=10.0,
            max_linear_m_s=0.03,
            max_angular_rad_s=0.10,
            goal_distance_tolerance_m=0.025,
            yaw_tolerance_rad=0.1,
        )

        self.assertIsInstance(result, FollowResult)
        self.assertFalse(result.reached_goal)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.end, Pose2D(0.0, 0.0, 0.0))
        self.assertEqual(robot.commands, [(0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
