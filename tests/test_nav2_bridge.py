from __future__ import annotations

import unittest

import numpy as np

from xlerobot_playground.maniskill_ros_bridge import build_parser, synthesize_scan_from_depth
from xlerobot_playground.nav2_params import patch_nav2_params, render_slam_toolbox_params


class Nav2BridgeTests(unittest.TestCase):
    def test_bridge_parser_accepts_nav2_relevant_topics(self) -> None:
        args = build_parser().parse_args(
            [
                "--env-id",
                "ReplicaCAD_SceneManipulation-v1",
                "--cmd-vel-topic",
                "/robot/cmd_vel",
                "--scan-topic",
                "/robot/scan",
                "--linear-cmd-gain",
                "1.5",
                "--angular-cmd-gain",
                "0.32",
                "--laser-max-range-m",
                "8.5",
                "--max-episode-steps",
                "1500",
                "--build-config-idx",
                "2",
                "--spawn-x",
                "-0.2",
                "--spawn-y",
                "-1.0",
                "--spawn-yaw",
                "1.57",
                "--ros-base-yaw-offset-rad",
                "3.14159",
                "--no-publish-head-camera",
            ]
        )
        self.assertEqual(args.env_id, "ReplicaCAD_SceneManipulation-v1")
        self.assertEqual(args.cmd_vel_topic, "/robot/cmd_vel")
        self.assertEqual(args.scan_topic, "/robot/scan")
        self.assertEqual(args.linear_cmd_gain, 1.5)
        self.assertEqual(args.angular_cmd_gain, 0.32)
        self.assertEqual(args.laser_max_range_m, 8.5)
        self.assertEqual(args.max_episode_steps, 1500)
        self.assertEqual(args.build_config_idx, 2)
        self.assertEqual(args.spawn_x, -0.2)
        self.assertEqual(args.spawn_y, -1.0)
        self.assertEqual(args.spawn_yaw, 1.57)
        self.assertEqual(args.ros_base_yaw_offset_rad, 3.14159)
        self.assertFalse(args.publish_head_camera)

    def test_depth_band_is_converted_into_laser_ranges(self) -> None:
        depth_mm = [[1000] * 5 for _ in range(7)]
        ranges, angles = synthesize_scan_from_depth(
            depth_mm=np.asarray(depth_mm, dtype="int16"),
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=10.0,
        )
        self.assertEqual(len(ranges), 5)
        self.assertEqual(len(angles), 5)
        self.assertTrue(all(value >= 1.0 for value in ranges[1:4]))
        self.assertGreater(float(angles[0]), 0.0)
        self.assertLess(float(angles[-1]), 0.0)

    def test_nav2_params_are_patched_to_bridge_topics(self) -> None:
        base = {
            "global_costmap": {
                "global_costmap": {
                    "ros__parameters": {
                        "plugins": ["static_layer", "obstacle_layer", "inflation_layer"],
                        "obstacle_layer": {},
                        "inflation_layer": {},
                    }
                }
            },
            "local_costmap": {
                "local_costmap": {
                    "ros__parameters": {
                        "plugins": ["voxel_layer", "inflation_layer"],
                        "voxel_layer": {},
                        "inflation_layer": {},
                    }
                }
            },
            "bt_navigator": {"ros__parameters": {}},
            "amcl": {"ros__parameters": {}},
            "behavior_server": {"ros__parameters": {}},
            "controller_server": {
                "ros__parameters": {
                    "FollowPath": {
                        "max_vel_x": 0.26,
                        "max_speed_xy": 0.26,
                        "max_vel_theta": 1.0,
                        "trans_stopped_velocity": 0.25,
                        "PathAlign.scale": 32.0,
                        "GoalAlign.scale": 24.0,
                        "RotateToGoal.scale": 32.0,
                        "RotateToGoal.slowing_factor": 5.0,
                    }
                }
            },
            "planner_server": {"ros__parameters": {}},
            "velocity_smoother": {
                "ros__parameters": {
                    "max_velocity": [0.26, 0.0, 1.0],
                    "min_velocity": [-0.26, 0.0, -1.0],
                }
            },
        }
        patched = patch_nav2_params(
            base,
            scan_topic="/bridge/scan",
            base_frame="base_link",
            odom_frame="odom",
            map_frame="map",
            robot_radius=0.3,
        )

        obstacle_scan = patched["global_costmap"]["global_costmap"]["ros__parameters"]["obstacle_layer"]["scan"]
        self.assertEqual(obstacle_scan["topic"], "/bridge/scan")
        self.assertEqual(patched["global_costmap"]["global_costmap"]["ros__parameters"]["robot_radius"], 0.3)
        self.assertEqual(patched["local_costmap"]["local_costmap"]["ros__parameters"]["global_frame"], "odom")
        self.assertEqual(
            patched["local_costmap"]["local_costmap"]["ros__parameters"]["voxel_layer"]["z_voxels"],
            32,
        )
        self.assertEqual(patched["bt_navigator"]["ros__parameters"]["odom_topic"], "/odom")
        self.assertEqual(patched["amcl"]["ros__parameters"]["base_frame_id"], "base_link")
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["max_vel_x"],
            0.65,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["max_vel_theta"],
            0.45,
        )
        self.assertEqual(
            patched["velocity_smoother"]["ros__parameters"]["max_velocity"][0],
            0.65,
        )
        self.assertEqual(
            patched["velocity_smoother"]["ros__parameters"]["max_velocity"][2],
            0.45,
        )
        self.assertEqual(
            patched["controller_server"]["ros__parameters"]["FollowPath"]["RotateToGoal.scale"],
            8.0,
        )

    def test_slam_toolbox_params_use_standard_frames(self) -> None:
        params = render_slam_toolbox_params()
        slam = params["slam_toolbox"]["ros__parameters"]
        self.assertEqual(slam["scan_topic"], "/scan")
        self.assertEqual(slam["map_frame"], "map")
        self.assertEqual(slam["odom_frame"], "odom")
        self.assertEqual(slam["base_frame"], "base_link")


if __name__ == "__main__":
    unittest.main()
