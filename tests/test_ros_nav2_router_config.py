from __future__ import annotations

import unittest

try:
    from xlerobot_playground.ros_nav2_router import build_fake_free_map, build_parser, config_from_args
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        build_fake_free_map = build_parser = config_from_args = None
    else:
        raise


@unittest.skipIf(build_parser is None, "local Python environment does not have numpy/ROS runtime dependencies")
class RosNav2RouterConfigTests(unittest.TestCase):

    def test_fake_free_map_arguments_are_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--fake-free-map", "--fake-map-size-m", "1.2", "--fake-map-resolution-m", "0.05"]
        )

        config = config_from_args(args)

        self.assertTrue(config.fake_free_map)
        self.assertEqual(config.fake_map_size_m, 1.2)
        self.assertEqual(config.fake_map_resolution_m, 0.05)

    def test_build_fake_free_map_is_centered_and_all_free(self) -> None:
        occupancy_map = build_fake_free_map(size_m=1.0, resolution_m=0.1)

        self.assertEqual(occupancy_map.width, 10)
        self.assertEqual(occupancy_map.height, 10)
        self.assertAlmostEqual(occupancy_map.origin_x, -0.5)
        self.assertAlmostEqual(occupancy_map.origin_y, -0.5)
        self.assertEqual(set(occupancy_map.data), {0})


if __name__ == "__main__":
    unittest.main()
