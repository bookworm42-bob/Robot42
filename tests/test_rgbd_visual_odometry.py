from __future__ import annotations

import math
import unittest

from xlerobot_playground.rgbd_visual_odometry import (
    PlanarPose,
    angle_wrap,
    build_parser,
    compose_planar,
    config_from_args,
    yaw_to_quaternion_xyzw,
)


class RgbdVisualOdometryHelperTests(unittest.TestCase):
    def test_compose_planar_forward(self) -> None:
        pose = compose_planar(PlanarPose(1.0, 2.0, math.pi / 2.0), 0.10, 0.0)

        self.assertAlmostEqual(pose.x, 1.0)
        self.assertAlmostEqual(pose.y, 2.10)
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0)

    def test_compose_planar_wraps_yaw(self) -> None:
        pose = compose_planar(PlanarPose(0.0, 0.0, math.radians(175.0)), 0.0, math.radians(20.0))

        self.assertAlmostEqual(pose.yaw, math.radians(-165.0))

    def test_yaw_quaternion_identity(self) -> None:
        x, y, z, w = yaw_to_quaternion_xyzw(0.0)

        self.assertEqual((x, y, z), (0.0, 0.0, 0.0))
        self.assertEqual(w, 1.0)

    def test_parser_config_converts_degrees(self) -> None:
        args = build_parser().parse_args(["--max-yaw-step-deg", "15", "--min-matches", "8"])
        config = config_from_args(args)

        self.assertEqual(config.min_matches, 8)
        self.assertAlmostEqual(config.max_yaw_step_rad, math.radians(15.0))

    def test_angle_wrap(self) -> None:
        self.assertAlmostEqual(angle_wrap(math.radians(181.0)), math.radians(-179.0))


if __name__ == "__main__":
    unittest.main()
