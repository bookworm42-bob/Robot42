from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig
from xlerobot_playground.sim_exploration_backend import (
    ManiSkillExplorationRunner,
    SimExplorationConfig,
    build_parser,
)


class SimExplorationBackendTests(unittest.TestCase):
    def test_parser_accepts_llm_policy_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "--session",
                "agentic_session",
                "--explorer-policy",
                "llm",
                "--llm-model",
                "gpt-4o-mini",
                "--sensor-range-m",
                "8.0",
                "--max-decisions",
                "12",
                "--nav2-planner-id",
                "SmacPlanner2D",
                "--nav2-controller-id",
                "FollowPath",
                "--serve-review-ui",
            ]
        )
        self.assertEqual(args.session, "agentic_session")
        self.assertEqual(args.explorer_policy, "llm")
        self.assertEqual(args.llm_model, "gpt-4o-mini")
        self.assertEqual(args.sensor_range_m, 8.0)
        self.assertEqual(args.max_decisions, 12)
        self.assertEqual(args.nav2_planner_id, "SmacPlanner2D")
        self.assertEqual(args.nav2_controller_id, "FollowPath")
        self.assertTrue(args.serve_review_ui)

    def test_runner_builds_real_agentic_map_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = str(Path(tmpdir) / "agentic_map.json")
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=persist_path,
                    occupancy_resolution=0.25,
                )
            )
            runner = ManiSkillExplorationRunner(
                SimExplorationConfig(
                    repo_root="/tmp/XLeRobot",
                    persist_path=persist_path,
                    area="apartment",
                    session="agentic_session",
                    source="test",
                    realtime_sleep_s=0.0,
                    llm_provider="mock",
                    llm_model="mock",
                    explorer_policy="llm",
                ),
                backend,
            )
            snapshot = runner.run()

            current_map = snapshot["current_map"]
            self.assertIsNotNone(current_map)
            assert current_map is not None
            self.assertEqual(current_map["mode"], "sim_agentic")
            self.assertEqual(current_map["map_id"], "agentic_session")
            self.assertGreater(current_map["coverage"], 0.75)
            self.assertGreaterEqual(len(current_map["regions"]), 4)
            self.assertTrue(current_map["trajectory"])
            self.assertTrue(current_map["keyframes"])
            self.assertTrue(current_map["occupancy"]["cells"])
            self.assertTrue(any(cell["state"] == "occupied" for cell in current_map["occupancy"]["cells"]))
            labels = {region["label"] for region in current_map["regions"]}
            self.assertIn("hallway", labels)
            self.assertIn("kitchen", labels)
            self.assertIn("living_room", labels)
            named_places = {item["name"] for item in current_map["named_places"]}
            self.assertIn("kitchen_center", named_places)
            self.assertIn("living_room_door", named_places)
            self.assertIn("hallway_entry", named_places)
            artifacts = current_map["artifacts"]
            self.assertTrue(artifacts["decision_log"])
            self.assertIn("frontier_memory", artifacts)
            self.assertIn("nav2", artifacts)
            self.assertEqual(artifacts["nav2"]["module"], "simulated_nav2")
            self.assertTrue(artifacts["nav2"]["goals"])
            self.assertTrue(Path(persist_path).exists())

            task = snapshot["active_task"]
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task["state"], "succeeded")
            self.assertGreaterEqual(task["result"]["decision_count"], 1)


if __name__ == "__main__":
    unittest.main()
