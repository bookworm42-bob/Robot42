from __future__ import annotations

import unittest

from xlerobot_agent.exploration import Pose2D
from xlerobot_playground.exploration_runtime import (
    NavigationPath,
    NavigationResult,
    ScanObservation,
)
from xlerobot_playground.runtime_exploration_session import RuntimeNeutralExplorationSession


class FakeRuntime:
    name = "fake_runtime"

    def __init__(self) -> None:
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.executed_paths = []
        self.stop_calls = 0

    def reset(self) -> None:
        self.pose = Pose2D(0.0, 0.0, 0.0)

    def current_pose(self) -> Pose2D:
        return self.pose

    def capture_rgbd(self):
        raise NotImplementedError("no rgbd")

    def latest_scan(self) -> ScanObservation:
        return ScanObservation(
            pose=self.pose,
            ranges=(2.0,),
            angle_min=0.0,
            angle_increment=0.1,
            range_min=0.05,
            range_max=4.0,
            frame_id="head_laser",
        )

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float) -> NavigationResult:
        return NavigationResult(succeeded=True, message="sent")

    def stop(self) -> NavigationResult:
        self.stop_calls += 1
        return NavigationResult(succeeded=True, message="stopped")

    def rotate_in_place(self, yaw_delta_rad: float) -> NavigationResult:
        return NavigationResult(succeeded=True, message="rotated")

    def execute_path(self, path) -> NavigationResult:
        self.executed_paths.append(tuple(path))
        return NavigationResult(succeeded=True, message="executed", final_pose=path[-1])


class FakeNavigation:
    def __init__(self) -> None:
        self.update_state_calls = []
        self.cancel_calls = 0

    def update_map(self, occupancy_map) -> None:
        pass

    def update_scan(self, scan) -> None:
        pass

    def update_state(self, *, occupancy_map=None, pose=None, scan=None, image_data_url=None) -> None:
        self.update_state_calls.append(
            {
                "occupancy_map": occupancy_map,
                "pose": pose,
                "scan": scan,
                "image_data_url": image_data_url,
            }
        )

    def plan_path(self, start: Pose2D, goal: Pose2D) -> NavigationPath:
        return NavigationPath(poses=(start, goal), source="fake_nav")

    def navigate_to_pose(self, goal: Pose2D) -> NavigationResult:
        return NavigationResult(succeeded=True, message="nav")

    def cancel(self) -> None:
        self.cancel_calls += 1


class RuntimeNeutralExplorationSessionTests(unittest.TestCase):
    def test_refresh_pushes_runtime_state_to_navigation_dependency(self) -> None:
        runtime = FakeRuntime()
        navigation = FakeNavigation()
        session = RuntimeNeutralExplorationSession(runtime, navigation)

        snapshot = session.refresh_navigation_state(occupancy_map="map")

        self.assertEqual(snapshot.runtime_name, "fake_runtime")
        self.assertEqual(snapshot.pose, Pose2D(0.0, 0.0, 0.0))
        self.assertEqual(navigation.update_state_calls[-1]["occupancy_map"], "map")
        self.assertEqual(navigation.update_state_calls[-1]["scan"].frame_id, "head_laser")
        self.assertIn("no rgbd", snapshot.errors)

    def test_navigate_to_plans_with_dependency_and_executes_on_runtime(self) -> None:
        runtime = FakeRuntime()
        navigation = FakeNavigation()
        session = RuntimeNeutralExplorationSession(runtime, navigation)

        result = session.navigate_to(Pose2D(1.0, 0.0, 0.0))

        self.assertTrue(result.succeeded)
        self.assertEqual(len(runtime.executed_paths), 1)
        self.assertEqual(runtime.executed_paths[0][-1], Pose2D(1.0, 0.0, 0.0))
        self.assertEqual(session.last_path.source, "fake_nav")

    def test_stop_cancels_navigation_and_stops_runtime(self) -> None:
        runtime = FakeRuntime()
        navigation = FakeNavigation()
        session = RuntimeNeutralExplorationSession(runtime, navigation)

        result = session.stop()

        self.assertTrue(result.succeeded)
        self.assertEqual(navigation.cancel_calls, 1)
        self.assertEqual(runtime.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
