from __future__ import annotations

import math
import unittest

from xlerobot_playground.real_exploration_runtime import (
    RealXLeRobotDirectRuntime,
    RealXLeRobotRuntimeConfig,
)


class FakeRobot:
    def __init__(self) -> None:
        self.connected = False
        self.actions = []
        self.stop_calls = 0

    def connect(self) -> None:
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

    def test_stop_uses_robot_stop_base_when_connected(self) -> None:
        robot = FakeRobot()
        runtime = RealXLeRobotDirectRuntime(robot=robot)
        runtime.connect()

        result = runtime.stop()

        self.assertTrue(result.succeeded)
        self.assertEqual(robot.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
