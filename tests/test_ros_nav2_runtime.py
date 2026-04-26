from __future__ import annotations

import math
import unittest

from xlerobot_playground.ros_nav2_runtime import compute_turn_command, remaining_turn_delta_rad


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


if __name__ == "__main__":
    unittest.main()
