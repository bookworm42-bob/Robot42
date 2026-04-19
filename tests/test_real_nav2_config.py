from __future__ import annotations

import unittest

from xlerobot_playground.real_nav2_config import build_parser


class RealNav2ConfigTests(unittest.TestCase):
    def test_parser_defaults_are_real_robot_safe(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.scan_topic, "/scan")
        self.assertEqual(args.map_frame, "map")
        self.assertEqual(args.odom_frame, "odom")
        self.assertEqual(args.base_frame, "base_link")
        self.assertEqual(args.max_linear_velocity, 0.03)
        self.assertEqual(args.max_angular_velocity, 0.10)


if __name__ == "__main__":
    unittest.main()
