from __future__ import annotations

import unittest

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import ScanObservation
from xlerobot_playground.nav2_navigation import (
    Nav2RouterNavigationDependency,
    path_length_m,
    scan_observation_to_payload,
)


class FakeRouterClient:
    def __init__(self) -> None:
        self.update_calls = []
        self.compute_calls = []

    def update_state(self, *, occupancy_map, pose, scan_observation, image_data_url):
        self.update_calls.append(
            {
                "occupancy_map": occupancy_map,
                "pose": pose,
                "scan_observation": scan_observation,
                "image_data_url": image_data_url,
            }
        )
        return {"status": "ok"}

    def compute_path(self, *, goal_pose, planner_id=""):
        self.compute_calls.append({"goal_pose": goal_pose, "planner_id": planner_id})
        return 4, [Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0), goal_pose], "succeeded"


class Nav2NavigationDependencyTests(unittest.TestCase):
    def test_scan_observation_to_payload_preserves_scan_geometry(self) -> None:
        scan = ScanObservation(
            pose=Pose2D(1.0, 2.0, 0.5),
            ranges=(1.0, 2.0),
            angle_min=-0.2,
            angle_increment=0.1,
            range_min=0.05,
            range_max=5.0,
            frame_id="head_laser",
        )

        payload = scan_observation_to_payload(scan)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["frame_id"], "head_laser")
        self.assertEqual(payload["pose"], Pose2D(1.0, 2.0, 0.5))
        self.assertEqual(payload["ranges"], (1.0, 2.0))

    def test_plan_path_publishes_runtime_state_then_calls_router(self) -> None:
        client = FakeRouterClient()
        dependency = Nav2RouterNavigationDependency(client, planner_id="GridBased")
        scan = ScanObservation(
            pose=Pose2D(0.0, 0.0, 0.0),
            ranges=(3.0,),
            angle_min=0.0,
            angle_increment=0.1,
            range_min=0.05,
            range_max=4.0,
            frame_id="head_laser",
        )

        dependency.update_state(occupancy_map="map", scan=scan, image_data_url="data:image/png;base64,test")
        path = dependency.plan_path(Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 1.0, 0.0))

        self.assertEqual(len(client.update_calls), 1)
        self.assertEqual(client.update_calls[0]["occupancy_map"], "map")
        self.assertEqual(client.update_calls[0]["pose"], Pose2D(0.0, 0.0, 0.0))
        self.assertEqual(client.update_calls[0]["scan_observation"]["frame_id"], "head_laser")
        self.assertEqual(client.compute_calls, [{"goal_pose": Pose2D(1.0, 1.0, 0.0), "planner_id": "GridBased"}])
        self.assertEqual(path.source, "ros_nav2_router")
        self.assertEqual(path.metadata["status_label"], "succeeded")
        self.assertAlmostEqual(path.cost_m, 2.0, places=3)

    def test_router_without_execution_reports_navigation_as_unsupported(self) -> None:
        dependency = Nav2RouterNavigationDependency(FakeRouterClient())

        result = dependency.navigate_to_pose(Pose2D(1.0, 0.0, 0.0))

        self.assertFalse(result.succeeded)
        self.assertIn("only supports path planning", result.message)

    def test_path_length(self) -> None:
        self.assertAlmostEqual(
            path_length_m([Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 4.0, 0.0)]),
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
