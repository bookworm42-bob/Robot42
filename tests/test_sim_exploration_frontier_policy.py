from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig, Pose2D
from xlerobot_playground.sim_exploration_backend import (
    ExplorationDecision,
    FrontierMemory,
    FrontierRecord,
    GridCell,
    ManiSkillExplorationRunner,
    SimExplorationConfig,
    build_parser,
)
from xlerobot_playground.interactive_exploration_playground import ManiSkillTeleportExplorationSession
from xlerobot_playground.interactive_exploration_playground import (
    _navigation_map_data_url,
    _resolve_manishkill_start_pose,
    _updated_mobile_base_qpos,
    _zero_mobile_base_qvel,
    build_parser as build_interactive_playground_parser,
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
                "--nav2-mode",
                "ros",
                "--nav2-planner-id",
                "SmacPlanner2D",
                "--nav2-controller-id",
                "FollowPath",
                "--ros-map-topic",
                "/map",
                "--serve-review-ui",
            ]
        )
        self.assertEqual(args.session, "agentic_session")
        self.assertEqual(args.explorer_policy, "llm")
        self.assertEqual(args.llm_model, "gpt-4o-mini")
        self.assertEqual(args.sensor_range_m, 8.0)
        self.assertEqual(args.max_decisions, 12)
        self.assertEqual(args.nav2_mode, "ros")
        self.assertEqual(args.nav2_planner_id, "SmacPlanner2D")
        self.assertEqual(args.nav2_controller_id, "FollowPath")
        self.assertEqual(args.ros_map_topic, "/map")
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
            self.assertIn("memory_updates", artifacts["decision_log"][0]["decision"])
            self.assertIn("applied_memory_updates", artifacts["decision_log"][0]["trace"])
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

    def test_manishkill_frontier_approach_pose_is_offset_to_known_side(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            robot_radius_m=0.22,
        )
        session.known_cells = {}
        session.guardrail_events = []
        for x in range(0, 8):
            for y in range(2, 8):
                session.known_cells[GridCell(x, y)] = "free"
        session.known_cells[GridCell(6, 5)] = "occupied"

        approach = session._select_frontier_approach_cell(
            cluster=[GridCell(5, 5)],
            unknown_cells={GridCell(6, 5)},
            robot_cell=GridCell(1, 5),
            reachable_safe_cells=session._reachable_safe_navigation_cells(GridCell(1, 5)),
        )

        self.assertIsNotNone(approach)
        assert approach is not None
        self.assertLess(approach.x, 5)
        self.assertTrue(session._is_valid_robot_center_cell(approach))

    def test_strict_robot_center_validation_rejects_unknown_footprint_space(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
            robot_radius_m=0.22,
        )
        session.known_cells = {}
        session.guardrail_events = []
        for x in range(0, 3):
            for y in range(0, 3):
                session.known_cells[GridCell(x, y)] = "free"

        self.assertFalse(
            session._is_valid_robot_center_cell(
                GridCell(0, 0),
                required_known_fraction=0.95,
                unknown_is_blocking=True,
                extra_clearance_m=0.08,
            )
        )

    def test_mobile_base_qpos_helpers_update_only_base_dofs(self) -> None:
        qpos = [0.0, 0.0, 0.0, 1.2, -0.5]
        qvel = [0.3, -0.1, 0.2, 0.9, -0.4]
        updated_qpos = _updated_mobile_base_qpos(
            qpos,
            Pose2D(1.5, -2.0, 0.7),
            anchor_pose=Pose2D(-1.0, 0.0, 0.0),
            anchor_qpos=[0.0, 0.0, 0.0, 1.2, -0.5],
        )
        updated_qvel = _zero_mobile_base_qvel(qvel)

        self.assertAlmostEqual(float(updated_qpos[0]), 2.5)
        self.assertAlmostEqual(float(updated_qpos[1]), -2.0)
        self.assertAlmostEqual(float(updated_qpos[2]), 0.7)
        self.assertEqual(list(updated_qpos[3:]), [1.2, -0.5])
        self.assertEqual(list(updated_qvel[:3]), [0.0, 0.0, 0.0])
        self.assertEqual(list(updated_qvel[3:]), [0.9, -0.4])

    def test_manishkill_candidate_paths_prefer_current_reachable_frontiers(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(2, 0): "free",
            GridCell(4, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(2, 0), GridCell(4, 0)}

        visible = FrontierRecord(
            frontier_id="frontier_visible",
            nav_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(2, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(4, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        inaccessible = FrontierRecord(
            frontier_id="frontier_inaccessible",
            nav_pose=GridCell(8, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(8, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=99,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {
            visible.frontier_id: visible,
            remembered.frontier_id: remembered,
            inaccessible.frontier_id: inaccessible,
        }

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_visible", "frontier_remembered"])
        self.assertIsNone(inaccessible.path_cost_m)

    def test_manishkill_candidate_paths_filter_frontier_at_current_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {GridCell(0, 0): "free", GridCell(3, 0): "free"}
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(3, 0)}

        current = FrontierRecord(
            frontier_id="frontier_current",
            nav_pose=GridCell(0, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(0, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=8,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        nearby = FrontierRecord(
            frontier_id="frontier_nearby",
            nav_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
        )
        session.frontier_memory.records = {
            current.frontier_id: current,
            nearby.frontier_id: nearby,
        }

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_nearby"])
        self.assertIsNone(current.path_cost_m)
        self.assertTrue(any(event["type"] == "frontier_at_current_pose_filtered" for event in session.guardrail_events))

    def test_manishkill_candidate_paths_do_not_expose_stored_memory_that_is_not_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(x, y): "free"
            for x in range(-6, 9)
            for y in range(-6, 7)
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(3, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(3, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertTrue(
            any(event["type"] == "stored_frontier_not_global_boundary_after_merge" for event in session.guardrail_events)
        )

    def test_manishkill_candidate_paths_resnap_stored_memory_frontier_to_reachable_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(2, 0): "free",
            GridCell(6, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0), GridCell(2, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(5, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_remembered"])
        self.assertEqual(session._world_to_cell(remembered.nav_pose.x, remembered.nav_pose.y), GridCell(2, 0))
        self.assertEqual(session._world_to_cell(remembered.centroid_pose.x, remembered.centroid_pose.y), GridCell(6, 0))
        self.assertIsNotNone(remembered.path_cost_m)
        self.assertTrue(
            any(event["type"] == "stored_frontier_revisit_pose_resnapped" for event in session.guardrail_events)
        )

    def test_manishkill_candidate_paths_keep_relaxed_stored_memory_boundary(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(3, 0): "free",
            GridCell(2, 0): "occupied",
            GridCell(4, 0): "occupied",
            GridCell(3, -1): "occupied",
            GridCell(6, 0): "free",
            GridCell(5, 0): "occupied",
            GridCell(7, 0): "occupied",
            GridCell(6, -1): "occupied",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(3, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_relaxed",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(5, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual([record.frontier_id for record in records], ["frontier_relaxed"])
        self.assertTrue(any("relaxed revalidation" in evidence for evidence in remembered.evidence))

    def test_manishkill_candidate_paths_drop_stored_memory_without_reachable_revisit_pose(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.known_cells = {
            GridCell(0, 0): "free",
            GridCell(6, 0): "free",
        }
        session.range_edge_cells = set()
        session.guardrail_events = []
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        remembered = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(10, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(6, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=5,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
        )
        session.frontier_memory.records = {remembered.frontier_id: remembered}

        records = session._refresh_candidate_paths()

        self.assertEqual(records, [])
        self.assertTrue(
            any(event["type"] == "stored_frontier_without_reachable_revisit_pose" for event in session.guardrail_events)
        )

    def test_manishkill_detects_single_edge_frontier_when_robot_sized_approach_exists(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {
            GridCell(0, 0): "free",
            GridCell(1, 0): "occupied",
            GridCell(-1, 0): "occupied",
            GridCell(0, 1): "occupied",
        }
        session.latest_scan_range_edge_cells = set()
        session.known_cells = dict(session.latest_scan_known_cells)
        session.range_edge_cells = set()
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        session._select_frontier_approach_cell = lambda **_kwargs: GridCell(0, 0)

        candidates = session._detect_frontier_candidates()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].unknown_gain, 1)
        self.assertTrue(any("single unknown-facing edge" in evidence for evidence in candidates[0].evidence))

    def test_manishkill_finish_guardrail_falls_back_to_reachable_stored_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.frontier_memory = FrontierMemory(session.config.occupancy_resolution)
        session.frontier_memory.return_waypoints = {}
        session.guardrail_events = []
        session._coverage = lambda: 0.4
        stored = FrontierRecord(
            frontier_id="frontier_remembered",
            nav_pose=GridCell(1, 0).center_pose(session.config.occupancy_resolution),
            centroid_pose=GridCell(1, 0).center_pose(session.config.occupancy_resolution),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=9,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
            path_cost_m=0.25,
        )

        class DummyPolicy:
            def _heuristic_decision(self, *_args: object) -> ExplorationDecision:
                return ExplorationDecision(
                    decision_type="revisit_frontier",
                    selected_frontier_id="frontier_remembered",
                    selected_return_waypoint_id=None,
                    frontier_ids_to_store=[],
                    exploration_complete=False,
                    reasoning_summary="Fallback to reachable stored frontier.",
                    semantic_updates=[],
                )

        session.policy = DummyPolicy()
        finish = ExplorationDecision(
            decision_type="finish",
            selected_frontier_id=None,
            selected_return_waypoint_id=None,
            frontier_ids_to_store=[],
            exploration_complete=True,
            reasoning_summary="No reachable frontier remains.",
            semantic_updates=[],
        )

        guarded = session._apply_finish_guardrail(finish, [stored])

        self.assertEqual(guarded.decision_type, "revisit_frontier")
        self.assertEqual(guarded.selected_frontier_id, "frontier_remembered")
        self.assertTrue(any(event["type"] == "finish_with_reachable_frontiers" for event in session.guardrail_events))

    def test_manishkill_rejects_latest_scan_boundary_that_is_not_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {GridCell(0, 0): "free"}
        session.latest_scan_range_edge_cells = set()
        session.known_cells = {
            GridCell(x, y): "free"
            for x in range(-1, 2)
            for y in range(-1, 2)
        }
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}

        candidates = session._detect_frontier_candidates()

        self.assertEqual(candidates, [])
        self.assertTrue(
            any(event["type"] == "frontier_not_global_boundary_after_merge" for event in session.guardrail_events)
        )

    def test_manishkill_keeps_latest_scan_boundary_that_remains_global_frontier(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session.config = SimExplorationConfig(
            repo_root="/tmp/XLeRobot",
            persist_path="",
            occupancy_resolution=0.25,
        )
        session.options = type("Options", (), {"max_frontiers": 12})()
        session.guardrail_events = []
        session.latest_scan_known_cells = {GridCell(0, 0): "free"}
        session.latest_scan_range_edge_cells = set()
        session.known_cells = {GridCell(0, 0): "free"}
        session._current_pose = lambda: Pose2D(0.125, 0.125, 0.0)
        session._reachable_safe_navigation_cells = lambda _cell: {GridCell(0, 0)}
        session._select_frontier_approach_cell = lambda **_kwargs: GridCell(0, 0)

        candidates = session._detect_frontier_candidates()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].unknown_gain, 4)
        self.assertTrue(
            any("passed merged-map validation" in evidence for evidence in candidates[0].evidence)
        )

    def test_navigation_map_data_url_renders_selectable_and_memory_frontiers(self) -> None:
        current = FrontierRecord(
            frontier_id="frontier_current",
            nav_pose=GridCell(2, 0).center_pose(0.25),
            centroid_pose=GridCell(2, 1).center_pose(0.25),
            status="stored",
            discovered_step=1,
            last_seen_step=2,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=True,
            path_cost_m=0.5,
        )
        remembered = FrontierRecord(
            frontier_id="frontier_memory",
            nav_pose=GridCell(0, 2).center_pose(0.25),
            centroid_pose=GridCell(0, 3).center_pose(0.25),
            status="stored",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=4,
            sensor_range_edge=False,
            room_hint=None,
            currently_visible=False,
            path_cost_m=1.0,
        )

        data_url = _navigation_map_data_url(
            known_cells={GridCell(0, 0): "free", GridCell(1, 0): "occupied"},
            resolution=0.25,
            trajectory=[Pose2D(0.125, 0.125, 0.0).to_dict()],
            robot_pose=Pose2D(0.125, 0.125, 0.0),
            candidate_records=[current],
            remembered_records=[remembered],
        )

        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_interactive_parser_accepts_spawn_facing_and_front_only_scan(self) -> None:
        args = build_interactive_playground_parser().parse_args(
            [
                "--backend",
                "maniskill",
                "--spawn-facing",
                "left",
                "--scan-mode",
                "front_only",
                "--scan-yaw-samples",
                "1",
            ]
        )

        self.assertEqual(args.backend, "maniskill")
        self.assertEqual(args.spawn_facing, "left")
        self.assertEqual(args.scan_mode, "front_only")
        self.assertEqual(args.scan_yaw_samples, 1)

    def test_resolve_manishkill_start_pose_rotates_default_spawn_in_place(self) -> None:
        pose = _resolve_manishkill_start_pose(
            Pose2D(-1.0, 0.0, 0.25),
            spawn_x=None,
            spawn_y=None,
            spawn_yaw=0.0,
            spawn_facing="left",
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.x, -1.0)
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.yaw, 0.25 - 1.5707963267948966, places=6)

    def test_resolve_manishkill_start_pose_keeps_absolute_custom_spawn_pose(self) -> None:
        pose = _resolve_manishkill_start_pose(
            Pose2D(-1.0, 0.0, 0.25),
            spawn_x=1.5,
            spawn_y=-0.75,
            spawn_yaw=0.6,
            spawn_facing="back",
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.x, 1.5)
        self.assertAlmostEqual(pose.y, -0.75)
        self.assertAlmostEqual(pose.yaw, 0.6)

    def test_display_pose_applies_heading_offset_without_moving_position(self) -> None:
        session = ManiSkillTeleportExplorationSession.__new__(ManiSkillTeleportExplorationSession)
        session._display_yaw_offset_rad = 3.141592653589793

        display_pose = session._display_pose(Pose2D(1.25, -0.75, 0.4))

        self.assertAlmostEqual(display_pose.x, 1.25)
        self.assertAlmostEqual(display_pose.y, -0.75)
        self.assertAlmostEqual(display_pose.yaw, -2.741592653589793, places=6)


if __name__ == "__main__":
    unittest.main()
