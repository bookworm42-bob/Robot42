from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from xlerobot_playground.real_exploration_runtime import (
    RealXLeRobotDirectRuntime,
    RealXLeRobotRuntimeConfig,
)


class FakeRobot:
    def __init__(self, *, connect_error: Exception | None = None) -> None:
        self.connected = False
        self.actions = []
        self.stop_calls = 0
        self.connect_error = connect_error

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def send_action(self, action):
        self.actions.append(dict(action))
        return dict(action)

    def stop_base(self) -> None:
        self.stop_calls += 1


class RealXLeRobotDirectRuntimeTests(unittest.TestCase):
    def test_motion_commands_are_disabled_by_default(self) -> None:
        robot = FakeRobot()
        runtime = RealXLeRobotDirectRuntime(robot=robot)

        result = runtime.drive_velocity(linear_m_s=0.1, angular_rad_s=0.2)

        self.assertFalse(result.succeeded)
        self.assertFalse(robot.connected)
        self.assertEqual(robot.actions, [])

    def test_velocity_command_is_clamped_and_converted_to_xlerobot_action(self) -> None:
        robot = FakeRobot()
        runtime = RealXLeRobotDirectRuntime(
            RealXLeRobotRuntimeConfig(
                robot_kind="xlerobot",
                allow_motion_commands=True,
                max_linear_m_s=0.2,
                max_angular_rad_s=0.5,
            ),
            robot=robot,
        )

        result = runtime.drive_velocity(linear_m_s=1.0, angular_rad_s=2.0)

        self.assertTrue(result.succeeded)
        self.assertTrue(robot.connected)
        self.assertEqual(len(robot.actions), 1)
        self.assertEqual(robot.actions[0]["x.vel"], 0.2)
        self.assertAlmostEqual(robot.actions[0]["theta.vel"], math.degrees(0.5))
        self.assertEqual(robot.actions[0]["y.vel"], 0.0)

    def test_two_wheel_action_omits_lateral_velocity(self) -> None:
        runtime = RealXLeRobotDirectRuntime(
            RealXLeRobotRuntimeConfig(robot_kind="xlerobot_2wheels"),
            robot=FakeRobot(),
        )

        action = runtime._base_velocity_action(linear_m_s=0.1, angular_rad_s=0.2)

        self.assertEqual(set(action), {"x.vel", "theta.vel"})

    def test_stop_sends_zero_velocity_action_when_connected(self) -> None:
        robot = FakeRobot()
        runtime = RealXLeRobotDirectRuntime(robot=robot)
        runtime.connect()

        result = runtime.stop()

        self.assertTrue(result.succeeded)
        self.assertEqual(robot.stop_calls, 0)
        self.assertEqual(robot.actions[-1]["x.vel"], 0.0)
        self.assertEqual(robot.actions[-1]["theta.vel"], 0.0)

    def test_connect_tolerates_already_connected_motor_bus(self) -> None:
        robot = FakeRobot(connect_error=RuntimeError("FeetechMotorsBus is already connected."))
        runtime = RealXLeRobotDirectRuntime(
            RealXLeRobotRuntimeConfig(allow_motion_commands=True),
            robot=robot,
        )

        result = runtime.drive_velocity(linear_m_s=0.01, angular_rad_s=0.0)

        self.assertTrue(result.succeeded)
        self.assertEqual(len(robot.actions), 1)

    def test_connect_auto_answers_calibration_prompt_when_configured(self) -> None:
        class PromptingRobot(FakeRobot):
            def connect(self) -> None:
                self.answer = input("restore?")
                self.connected = True

        robot = PromptingRobot()
        runtime = RealXLeRobotDirectRuntime(
            RealXLeRobotRuntimeConfig(calibration_prompt_response=""),
            robot=robot,
        )

        runtime.connect()

        self.assertEqual(robot.answer, "")

    def test_connect_leaves_input_interactive_when_prompt_response_is_none(self) -> None:
        robot = FakeRobot()
        runtime = RealXLeRobotDirectRuntime(
            RealXLeRobotRuntimeConfig(calibration_prompt_response=None),
            robot=robot,
        )

        with patch("builtins.input") as mocked_input:
            runtime.connect()

        mocked_input.assert_not_called()


if __name__ == "__main__":
    unittest.main()
