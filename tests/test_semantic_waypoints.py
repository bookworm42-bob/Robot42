from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from xlerobot_agent.exploration import Pose2D
from xlerobot_agent.semantic_prompts import (
    build_semantic_consolidation_system_prompt,
    build_semantic_evidence_extraction_system_prompt,
)
from xlerobot_playground.semantic_anchors import build_semantic_anchor_candidate
from xlerobot_playground.semantic_evidence import (
    PixelRegion,
    SemanticEvidence,
    parse_semantic_observation_payload,
)
from xlerobot_playground.semantic_memory import SemanticMemory
from xlerobot_playground.semantic_projection import CameraIntrinsics, project_pixel_region_to_map
from xlerobot_playground.sim_exploration_backend import (
    ExplorationLLMPolicy,
    FrontierRecord,
    SimExplorationConfig,
)


class _FakeRouter:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.model_suite = SimpleNamespace(planner=object())

    def complete_json_messages(self, *, config: object, messages: list[dict]) -> tuple[dict, SimpleNamespace]:
        return self.payload, SimpleNamespace(duration_s=0.0, error=None, response_text="{}")


class SemanticWaypointTests(unittest.TestCase):
    def test_semantic_observation_validation_rejects_map_coordinates(self) -> None:
        observations, warnings = parse_semantic_observation_payload(
            {
                "frame_id": "kf_001",
                "semantic_observations": [
                    {
                        "label_hint": "kitchen",
                        "confidence": 0.9,
                        "x": 1.2,
                        "pixel_regions": [
                            {
                                "bbox_xyxy": [0, 0, 10, 10],
                                "representative_point_uv": [5, 5],
                                "description": "counter",
                            }
                        ],
                    },
                    {
                        "label_hint": "dining area",
                        "confidence": 0.7,
                        "visual_cues": ["table", "chairs"],
                        "pixel_regions": [
                            {
                                "bbox_xyxy": [10, 20, 80, 100],
                                "representative_point_uv": [40, 60],
                                "image_position": "center",
                                "object_label": "table",
                                "description": "table and chairs",
                            }
                        ],
                    },
                ],
            }
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].label_hint, "dining_area")
        self.assertTrue(any("forbidden map coordinate" in item for item in warnings))

    def test_pixel_projection_uses_depth_and_camera_yaw(self) -> None:
        region = PixelRegion(
            frame_id="kf_001",
            bbox_xyxy=(0, 0, 4, 4),
            center_uv=(2, 2),
            depth_m=None,
            image_position="center",
            object_label="counter",
            description="counter",
        )
        pose = project_pixel_region_to_map(
            pixel_region=region,
            depth_image=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
            ],
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0, width=4, height=3),
            camera_pose=Pose2D(1.0, 1.0, math.pi / 2.0),
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.x, 1.0, places=3)
        self.assertAlmostEqual(pose.y, 3.0, places=3)

    def test_anchor_selection_rejects_when_no_known_free_reachable_pose_exists(self) -> None:
        evidence = SemanticEvidence(
            evidence_id="sem_ev_000001",
            label_hint="kitchen",
            evidence_pose=Pose2D(2.0, 2.0, 0.0),
            source_frame_ids=("kf_001",),
            source_pixels=tuple(),
            confidence=0.7,
            evidence=("counter visible",),
        )

        anchor = build_semantic_anchor_candidate(
            anchor_id="sem_anchor_000001",
            evidence=evidence,
            known_cells={(0, 0): "free", (1, 0): "occupied"},
            resolution=1.0,
            robot_cell=(0, 0),
            min_radius_m=0.5,
            max_radius_m=1.5,
        )

        self.assertEqual(anchor.reachability_status, "unreachable")
        self.assertEqual(anchor.status, "rejected")

    def test_semantic_memory_merges_duplicate_named_places(self) -> None:
        memory = SemanticMemory()
        first = SemanticEvidence(
            evidence_id="sem_ev_000001",
            label_hint="living room",
            evidence_pose=Pose2D(1.0, 1.0, 0.0),
            source_frame_ids=("kf_001",),
            source_pixels=tuple(),
            confidence=0.6,
            evidence=("sofa visible",),
        )
        second = SemanticEvidence(
            evidence_id="sem_ev_000002",
            label_hint="lounge",
            evidence_pose=Pose2D(1.2, 1.1, 0.0),
            source_frame_ids=("kf_002",),
            source_pixels=tuple(),
            confidence=0.8,
            evidence=("tv visible",),
        )
        memory.add_evidence(first)
        cluster = memory.add_evidence(second)

        self.assertEqual(len(memory.clusters), 1)
        self.assertEqual(cluster.label_hint, "living_room")
        self.assertEqual(set(cluster.evidence_ids), {"sem_ev_000001", "sem_ev_000002"})

    def test_frontier_policy_drops_semantic_updates_by_default(self) -> None:
        policy = ExplorationLLMPolicy(
            SimExplorationConfig(repo_root="/tmp/XLeRobot", persist_path="", explorer_policy="heuristic")
        )
        record = FrontierRecord(
            frontier_id="frontier_001",
            nav_pose=Pose2D(1.0, 0.0, 0.0),
            centroid_pose=Pose2D(1.2, 0.0, 0.0),
            status="candidate",
            discovered_step=1,
            last_seen_step=1,
            unknown_gain=10,
            sensor_range_edge=False,
            room_hint="region_kitchen",
            evidence=["frontier near kitchen-like room"],
            path_cost_m=1.0,
            currently_visible=True,
        )

        decision = policy._heuristic_decision([record], [], coverage=0.2, current_room_id="region_hallway")

        self.assertEqual(decision.semantic_updates, [])

    def test_frontier_policy_traces_ignored_legacy_semantic_updates(self) -> None:
        policy = ExplorationLLMPolicy(
            SimExplorationConfig(repo_root="/tmp/XLeRobot", persist_path="", explorer_policy="llm")
        )
        policy.router = _FakeRouter(
            {
                "decision_type": "finish",
                "selected_frontier_id": None,
                "selected_return_waypoint_id": None,
                "frontier_ids_to_store": [],
                "memory_updates": [],
                "exploration_complete": True,
                "reasoning_summary": "No frontiers remain.",
                "semantic_updates": [
                    {"label": "kitchen", "kind": "room_hint", "target_id": "frontier_001", "confidence": 0.9}
                ],
            }
        )

        decision, trace = policy.decide(
            prompt_payload={},
            frontiers=[],
            return_waypoints=[],
            coverage=1.0,
            current_room_id=None,
        )

        self.assertEqual(decision.semantic_updates, [])
        self.assertIn("ignored_legacy_frontier_semantic_update", trace)
        self.assertEqual(trace["ignored_legacy_frontier_semantic_update"]["count"], 1)

    def test_semantic_prompts_forbid_invented_coordinates(self) -> None:
        evidence_prompt = build_semantic_evidence_extraction_system_prompt()
        consolidation_prompt = build_semantic_consolidation_system_prompt()

        self.assertIn("Do not return map coordinates", evidence_prompt)
        self.assertIn("deterministic RGB-D projection", evidence_prompt)
        self.assertIn("do not invent map coordinates", consolidation_prompt)
        self.assertIn("reachable anchor", consolidation_prompt)


if __name__ == "__main__":
    unittest.main()
