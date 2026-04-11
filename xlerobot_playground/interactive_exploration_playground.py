from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time
import webbrowser
from typing import Any

import numpy as np

from xlerobot_agent.exploration import Pose2D
from xlerobot_agent.prompts import build_exploration_policy_user_prompt
from xlerobot_playground.maniskill_ros_bridge import (
    HEAD_CAMERA_FOV_RAD,
    HEAD_CAMERA_UID,
    normalize_quaternion_wxyz,
    quaternion_to_yaw,
    synthesize_scan_from_depth,
)
from xlerobot_playground.sim_exploration_backend import (
    ExplorationDecision,
    ExplorationLLMPolicy,
    FrontierCandidate,
    FrontierMemory,
    FrontierRecord,
    GridCell,
    SimExplorationConfig,
    _aggregate_semantic_updates,
    _build_simple_apartment,
    _centroid_cell,
    _dedupe_text,
    _frontier_selection_guidance,
    _grid_distance_cells,
    _neighbors4,
    _neighbors8,
    _pose_distance_m,
    _simulate_scan,
)


XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M = 0.3913
XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M = 0.459
XLEROBOT_IKEA_CART_FOOTPRINT_RADIUS_M = math.hypot(
    XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M / 2.0,
    XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M / 2.0,
)
XLEROBOT_IKEA_CART_CLEARANCE_PADDING_M = 0.06
BASE_TELEPORT_POSITION_TOLERANCE_M = 0.08
BASE_TELEPORT_YAW_TOLERANCE_RAD = 0.15
STRICT_NAVIGATION_CLEARANCE_MARGIN_M = 0.08
STRICT_NAVIGATION_KNOWN_FRACTION = 0.95
STORED_FRONTIER_REVALIDATION_RADIUS_M = 1.0
STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M = 1.25


@dataclass(frozen=True)
class ManiSkillInteractiveOptions:
    repo_root: str
    env_id: str
    robot_uid: str
    display_yaw_offset_deg: float | None
    control_mode: str
    render_mode: str | None
    shader: str
    sim_backend: str
    num_envs: int
    force_reload: bool
    build_config_idx: int | None
    spawn_x: float | None
    spawn_y: float | None
    spawn_yaw: float
    spawn_facing: str
    scan_mode: str
    scan_yaw_samples: int
    depth_beam_stride: int
    teleport_z: float | None
    teleport_settle_steps: int
    max_frontiers: int


class InteractiveNoNav2ExplorationSession:
    """Step-gated exploration sandbox for testing LLM frontier decisions without Nav2."""

    def __init__(self, config: SimExplorationConfig) -> None:
        self.config = config
        self.scenario = _build_simple_apartment(config.occupancy_resolution)
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.frontier_memory = FrontierMemory(self.config.occupancy_resolution)
            self.current_cell = self.scenario.start_cell
            self.current_yaw = 0.0
            self.known_cells: dict[GridCell, str] = {}
            self.range_edge_cells: set[GridCell] = set()
            self.trajectory: list[dict[str, Any]] = [
                self.current_cell.center_pose(self.scenario.resolution).to_dict()
            ]
            self.keyframes: list[dict[str, Any]] = []
            self.room_frames: dict[str, list[str]] = {}
            self.room_objects_seen: dict[str, set[str]] = {room_id: set() for room_id in self.scenario.rooms}
            self.room_descriptions: dict[str, list[str]] = {room_id: [] for room_id in self.scenario.rooms}
            self.decision_log: list[dict[str, Any]] = []
            self.semantic_updates: list[dict[str, Any]] = []
            self.guardrail_events: list[dict[str, Any]] = []
            self.total_distance_m = 0.0
            self.control_steps = 0
            self.decision_index = 0
            self.status = "initial_scan_complete"
            self.pending_prompt_payload: dict[str, Any] | None = None
            self.pending_prompt_text: str | None = None
            self.pending_candidate_records: list[FrontierRecord] = []
            self.pending_decision: ExplorationDecision | None = None
            self.pending_trace: dict[str, Any] | None = None
            self.applied_memory_updates: list[dict[str, Any]] = []
            self.last_error: str | None = None
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=self.current_cell.center_pose(self.scenario.resolution),
                step_index=0,
                reason="initial_pose",
            )
            self._perform_scan(full_turnaround=True, capture_frame=True, reason="mock_initial_turnaround_scan")
            self._prepare_decision_locked()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": round(self._coverage(), 3),
                "robot_pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [record.to_dict() for record in self.pending_candidate_records],
                "pending_decision": None if self.pending_decision is None else self.pending_decision.to_dict(),
                "pending_trace": self.pending_trace,
                "pending_target": self._pending_target(),
                "applied_memory_updates": list(self.applied_memory_updates),
                "last_error": self.last_error,
                "map": self._build_map_payload(),
            }

    def call_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.pending_prompt_payload is None:
                self._prepare_decision_locked()
            assert self.pending_prompt_payload is not None
            coverage = self._coverage()
            decision, trace = self.policy.decide(
                prompt_payload=self.pending_prompt_payload,
                frontiers=self.pending_candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=coverage,
                current_room_id=self.scenario.room_for_cell(self.current_cell),
            )
            decision = self._apply_finish_guardrail(decision, self.pending_candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.pending_decision = decision
            self.pending_trace = trace
            self.applied_memory_updates = applied_memory_updates
            self.status = "llm_response_ready"
            return self.snapshot()

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision. Click `Call LLM` first."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self._log_decision("finish_without_motion")
                self.pending_decision = None
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
            target_cell = self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y)
            if not self.scenario.in_bounds(target_cell) or target_cell not in self._known_free_cells():
                self.frontier_memory.fail(record.frontier_id, "mock mover target is not known free space")
                self.last_error = f"Mock mover rejected `{record.frontier_id}` because the target is not known free."
                self._log_decision("mock_move_rejected")
                self._prepare_decision_locked()
                return self.snapshot()

            self.current_yaw = math.atan2(target_cell.y - self.current_cell.y, target_cell.x - self.current_cell.x)
            self.current_cell = target_cell
            reached_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
            distance_m = _pose_distance_m(start_pose, reached_pose)
            self.total_distance_m += distance_m
            self.control_steps += 1
            self.trajectory.append(reached_pose.to_dict())
            self._perform_scan(
                full_turnaround=True,
                capture_frame=True,
                reason=f"mock_arrive_frontier::{record.frontier_id}",
            )
            self.frontier_memory.complete(record.frontier_id)
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=reached_pose,
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_decision("mock_move_applied")
            self.pending_decision = None
            self.pending_trace = None
            self.applied_memory_updates = []
            if self._coverage() >= self.config.finish_coverage_threshold:
                self.status = "finished"
            else:
                self._prepare_decision_locked()
            return self.snapshot()

    def _prepare_decision_locked(self) -> None:
        self.decision_index += 1
        visible_candidates = self._detect_frontier_candidates()
        self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
        candidate_records = self._refresh_candidate_paths()
        prompt_payload = self._build_prompt_payload(candidate_records)
        self.pending_candidate_records = candidate_records
        self.pending_prompt_payload = prompt_payload
        self.pending_prompt_text = build_exploration_policy_user_prompt(prompt_payload)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        self.status = "waiting_for_llm"
        if not candidate_records:
            self.status = "finished"

    def _perform_scan(self, *, full_turnaround: bool, capture_frame: bool, reason: str) -> None:
        scan = _simulate_scan(
            self.scenario,
            self.current_cell,
            yaw=self.current_yaw,
            max_range_m=self.config.sensor_range_m,
            full_turnaround=full_turnaround,
        )
        for cell in scan.observed_free:
            self.known_cells[cell] = "free"
        for cell in scan.observed_occupied:
            self.known_cells[cell] = "occupied"
        self.range_edge_cells |= scan.range_edge_frontiers
        current_room_id = self.scenario.room_for_cell(self.current_cell)
        for label in scan.visible_objects:
            if current_room_id:
                self.room_objects_seen.setdefault(current_room_id, set()).add(label)
        if current_room_id and scan.description not in self.room_descriptions.setdefault(current_room_id, []):
            self.room_descriptions[current_room_id].append(scan.description)
        if capture_frame:
            frame_id = f"kf_{len(self.keyframes) + 1:03d}"
            frame = {
                "frame_id": frame_id,
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "region_id": current_room_id or "unknown",
                "visible_objects": list(scan.visible_objects),
                "point_count": scan.point_count,
                "depth_min_m": scan.depth_min_m,
                "depth_max_m": scan.depth_max_m,
                "description": scan.description,
                "thumbnail_data_url": scan.thumbnail_data_url,
            }
            self.keyframes.append(frame)
            if current_room_id:
                self.room_frames.setdefault(current_room_id, []).append(frame_id)

    def _known_free_cells(self) -> set[GridCell]:
        return {cell for cell, state in self.known_cells.items() if state == "free"}

    def _detect_frontier_candidates(self) -> list[FrontierCandidate]:
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        known_free = self._known_free_cells()
        for cell in known_free:
            unknown_neighbors = {
                neighbor
                for neighbor in _neighbors4(cell)
                if self.scenario.in_bounds(neighbor) and neighbor not in self.known_cells
            }
            if unknown_neighbors:
                frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        candidates: list[FrontierCandidate] = []
        for cluster in clusters:
            cluster_unknown = set().union(*(unknown_neighbors_by_frontier.get(cell, set()) for cell in cluster))
            if not cluster_unknown:
                continue
            max_unknown_neighbor_count = max(len(unknown_neighbors_by_frontier.get(cell, set())) for cell in cluster)
            nav_cell = min(cluster, key=lambda cell: _grid_distance_cells(cell, self.current_cell))
            centroid_cell = _centroid_cell(cluster)
            room_hint = self.scenario.room_for_cell(nav_cell)
            evidence = [
                f"{len(cluster_unknown)} unknown neighbor cells",
                f"cluster size {len(cluster)}",
                "frontier information from partial RGB-D scan, not complete apartment knowledge",
            ]
            if max_unknown_neighbor_count >= 2:
                evidence.append("frontier signal is stronger: multiple unknown-facing neighbors support possible expansion")
            else:
                evidence.append(
                    "frontier signal is weaker: a single unknown-facing edge can still be useful, but the LLM should veto it if RGB/map context shows no meaningful navigable expansion"
                )
            if any(cell in self.range_edge_cells for cell in cluster):
                evidence.append("visible frontier reaches sensor range limit")
            candidates.append(
                FrontierCandidate(
                    frontier_id=None,
                    member_cells=tuple(sorted(cluster)),
                    nav_cell=nav_cell,
                    centroid_cell=centroid_cell,
                    nav_pose=nav_cell.center_pose(self.scenario.resolution),
                    centroid_pose=centroid_cell.center_pose(self.scenario.resolution),
                    unknown_gain=len(cluster_unknown),
                    sensor_range_edge=any(cell in self.range_edge_cells for cell in cluster),
                    room_hint=room_hint,
                    evidence=evidence,
                )
            )
        return candidates

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        current_pose = self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw)
        updated: list[FrontierRecord] = []
        for record in self.frontier_memory.candidate_records():
            if self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y) not in self._known_free_cells():
                record.path_cost_m = None
            else:
                record.path_cost_m = _pose_distance_m(current_pose, record.nav_pose)
            updated.append(record)
        updated.sort(
            key=lambda record: (
                record.path_cost_m is None,
                record.path_cost_m if record.path_cost_m is not None else 1e9,
                -(record.llm_memory_priority or 0.0),
                -record.unknown_gain,
            )
        )
        return updated

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        frontier_memory_snapshot = self.frontier_memory.snapshot()
        explored_areas = []
        for room_id, room in self.scenario.rooms.items():
            explored_areas.append(
                {
                    "region_id": room_id,
                    "label": room.label,
                    "observed_fraction": round(self._room_coverage(room_id), 3),
                    "objects_seen": sorted(self.room_objects_seen.get(room_id, set())),
                    "representative_frames": self.room_frames.get(room_id, [])[:3],
                }
            )
        return {
            "mission": (
                "Interactive no-Nav2 test: evaluate whether the LLM can use partial RGB-D frontier information, "
                "recent RGB views, and the 2D occupancy map to choose useful robot-navigable exploration regions."
            ),
            "robot": {
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "room_id": self.scenario.room_for_cell(self.current_cell),
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
                "navigation_mode": "mock_direct_move_no_nav2",
            },
            "frontier_memory": frontier_memory_snapshot,
            "frontier_information": [record.to_dict() for record in candidate_records],
            "candidate_frontiers": [record.to_dict() for record in candidate_records],
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "explored_areas": explored_areas,
            "recent_views": self.keyframes[-3:],
            "guardrails": {
                "frontier_information_is_partial_rgbd_scan_evidence": True,
                "navigation_is_mocked_without_nav2": True,
                "frontier_ids_must_come_from_prompt": True,
                "memory_updates_must_use_existing_frontier_ids": True,
                "avoid_furniture_shadow_boundaries_without_clear_open_space": True,
            },
            "ascii_map": self._ascii_map(candidate_records),
        }

    def _ascii_map(self, candidate_records: list[FrontierRecord]) -> str:
        frontier_cells = {
            self.scenario.world_to_cell(record.nav_pose.x, record.nav_pose.y): record.status
            for record in candidate_records
        }
        lines: list[str] = []
        for y in reversed(range(self.scenario.height_cells)):
            row: list[str] = []
            for x in range(self.scenario.width_cells):
                cell = GridCell(x, y)
                if cell == self.current_cell:
                    row.append("R")
                    continue
                if cell in frontier_cells:
                    row.append("V" if frontier_cells[cell] in {"completed", "failed", "suppressed"} else "F")
                    continue
                state = self.known_cells.get(cell)
                if state == "free":
                    row.append(".")
                elif state == "occupied":
                    row.append("#")
                else:
                    row.append("?")
            lines.append("".join(row))
        return "\n".join(lines)

    def _apply_finish_guardrail(
        self,
        decision: ExplorationDecision,
        candidate_records: list[FrontierRecord],
    ) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable and self._coverage() < self.config.finish_coverage_threshold:
            fallback = self.policy._heuristic_decision(
                candidate_records,
                list(self.frontier_memory.return_waypoints.values()),
                self._coverage(),
                self.scenario.room_for_cell(self.current_cell),
            )
            self.guardrail_events.append(
                {
                    "type": "finish_override",
                    "requested": decision.to_dict(),
                    "fallback": fallback.to_dict(),
                }
            )
            return fallback
        return decision

    def _log_decision(self, event_type: str) -> None:
        if self.pending_decision is None:
            return
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "event_type": event_type,
                "coverage": round(self._coverage(), 3),
                "decision": self.pending_decision.to_dict(),
                "trace": self.pending_trace or {},
                "applied_memory_updates": list(self.applied_memory_updates),
            }
        )

    def _pending_target(self) -> dict[str, Any] | None:
        if self.pending_decision is None or not self.pending_decision.selected_frontier_id:
            return None
        record = self.frontier_memory.records.get(self.pending_decision.selected_frontier_id)
        if record is None:
            return None
        return {
            "frontier_id": record.frontier_id,
            "nav_pose": record.nav_pose.to_dict(),
            "centroid_pose": record.centroid_pose.to_dict(),
            "status": record.status,
            "path_cost_m": record.path_cost_m,
        }

    def _coverage(self) -> float:
        discovered_free = sum(1 for cell in self.scenario.free_cells if self.known_cells.get(cell) == "free")
        return round(discovered_free / max(self.scenario.total_free_cells(), 1), 6)

    def _room_coverage(self, room_id: str) -> float:
        room_cells = [cell for cell, observed_room in self.scenario.room_by_cell.items() if observed_room == room_id]
        if not room_cells:
            return 0.0
        known = sum(1 for cell in room_cells if self.known_cells.get(cell) == "free")
        return known / len(room_cells)

    def _build_map_payload(self) -> dict[str, Any]:
        occupancy_cells = []
        for cell, state in sorted(self.known_cells.items()):
            occupancy_cells.append(
                {
                    "x": round(cell.x * self.scenario.resolution, 3),
                    "y": round(cell.y * self.scenario.resolution, 3),
                    "state": state,
                }
            )
        semantic_area_candidates = _aggregate_semantic_updates(self.semantic_updates)
        return {
            "map_id": self.config.session,
            "frame": "map",
            "resolution": float(self.scenario.resolution),
            "coverage": round(self._coverage(), 3),
            "mode": "interactive_no_nav2_llm_frontier",
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": [],
            "frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
            "semantic_area_candidates": semantic_area_candidates,
            "occupancy": {
                "resolution": float(self.scenario.resolution),
                "bounds": self.scenario.bounds(),
                "cells": occupancy_cells,
            },
            "artifacts": {
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "guardrail_events": self.guardrail_events,
                "navigation": {
                    "mode": "mock_direct_move_no_nav2",
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }


class ManiSkillTeleportExplorationSession:
    """Interactive no-Nav2 session backed by the real ManiSkill scene and XLeRobot RGB-D."""

    def __init__(self, config: SimExplorationConfig, options: ManiSkillInteractiveOptions) -> None:
        self.config = config
        self.options = options
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self._lock = threading.RLock()
        display_offset_deg = 0.0 if options.display_yaw_offset_deg is None else float(options.display_yaw_offset_deg)
        self._display_yaw_offset_rad = math.radians(display_offset_deg)
        self._initialize_environment()
        self.reset()

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def pump_viewer(self) -> None:
        if self.options.render_mode != "human":
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self.env.render()
        finally:
            self._lock.release()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_environment()
            self.frontier_memory = FrontierMemory(self.config.occupancy_resolution)
            self.known_cells: dict[GridCell, str] = {}
            self.range_edge_cells: set[GridCell] = set()
            self.latest_scan_cells: set[GridCell] = set()
            self.latest_scan_known_cells: dict[GridCell, str] = {}
            self.latest_scan_range_edge_cells: set[GridCell] = set()
            self.trajectory: list[dict[str, Any]] = [self._current_pose().to_dict()]
            self.keyframes: list[dict[str, Any]] = []
            self.decision_log: list[dict[str, Any]] = []
            self.semantic_updates: list[dict[str, Any]] = []
            self.guardrail_events: list[dict[str, Any]] = []
            self.total_distance_m = 0.0
            self.control_steps = 0
            self.decision_index = 0
            self.status = "initial_scan_complete"
            self.pending_prompt_payload: dict[str, Any] | None = None
            self.pending_prompt_text: str | None = None
            self.pending_candidate_records: list[FrontierRecord] = []
            self.pending_decision: ExplorationDecision | None = None
            self.pending_trace: dict[str, Any] | None = None
            self.applied_memory_updates: list[dict[str, Any]] = []
            self.last_error: str | None = None
            self.frontier_memory.remember_return_waypoint(
                room_id=None,
                pose=self._current_pose(),
                step_index=0,
                reason="initial_pose",
            )
            self._perform_scan(
                full_turnaround=self._scan_uses_turnaround(),
                capture_frame=True,
                reason=self._scan_reason("initial"),
            )
            self._prepare_decision_locked()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            raw_pose = self._current_pose()
            return {
                "status": self.status,
                "session": self.config.session,
                "coverage": round(self._coverage(), 3),
                "robot_pose": self._display_pose(raw_pose).to_dict(),
                "robot_raw_pose": raw_pose.to_dict(),
                "prompt": self.pending_prompt_text,
                "prompt_payload": self.pending_prompt_payload,
                "candidate_frontiers": [record.to_dict() for record in self.pending_candidate_records],
                "pending_decision": None if self.pending_decision is None else self.pending_decision.to_dict(),
                "pending_trace": self.pending_trace,
                "pending_target": self._pending_target(),
                "applied_memory_updates": list(self.applied_memory_updates),
                "last_error": self.last_error,
                "map": self._build_map_payload(),
            }

    def call_llm(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.pending_prompt_payload is None:
                self._prepare_decision_locked()
            assert self.pending_prompt_payload is not None
            decision_index = self.decision_index
            prompt_payload = json.loads(json.dumps(self.pending_prompt_payload))
            candidate_records = list(self.pending_candidate_records)
            return_waypoints = list(self.frontier_memory.return_waypoints.values())
            coverage = self._coverage()

        decision, trace = self.policy.decide(
            prompt_payload=prompt_payload,
            frontiers=candidate_records,
            return_waypoints=return_waypoints,
            coverage=coverage,
            current_room_id=None,
        )

        with self._lock:
            if decision_index != self.decision_index:
                self.last_error = "Discarded stale LLM response because the session changed while the model was running."
                return self.snapshot()
            decision = self._apply_finish_guardrail(decision, self.pending_candidate_records)
            applied_memory_updates = self.frontier_memory.apply_model_memory_updates(
                decision.memory_updates,
                selected_frontier_id=decision.selected_frontier_id,
            )
            trace["applied_memory_updates"] = applied_memory_updates
            self.pending_decision = decision
            self.pending_trace = trace
            self.applied_memory_updates = applied_memory_updates
            self.status = "llm_response_ready"
            return self.snapshot()

    def apply_decision(self) -> dict[str, Any]:
        with self._lock:
            self.last_error = None
            if self.pending_decision is None:
                self.last_error = "No pending LLM decision. Click `Call LLM` first."
                return self.snapshot()
            decision = self.pending_decision
            if decision.decision_type == "finish" or decision.exploration_complete:
                self.status = "finished"
                self._log_decision("finish_without_motion")
                self.pending_decision = None
                return self.snapshot()
            if not decision.selected_frontier_id:
                self.last_error = "The LLM decision did not select a frontier."
                return self.snapshot()
            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.last_error = f"Selected frontier `{decision.selected_frontier_id}` no longer exists."
                return self.snapshot()

            start_pose = self._current_pose()
            current_cell = self._world_to_cell(start_pose.x, start_pose.y)
            reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)
            target_cell = self._world_to_cell(record.nav_pose.x, record.nav_pose.y)
            if target_cell not in reachable_safe_cells:
                self.frontier_memory.fail(
                    record.frontier_id,
                    "teleport target is not in robot-connected, footprint-eroded free space",
                )
                self.last_error = (
                    f"Teleport rejected `{record.frontier_id}` because the target is not in "
                    "robot-connected, footprint-eroded free space."
                )
                self._log_decision("teleport_rejected")
                self._prepare_decision_locked()
                return self.snapshot()

            target_pose = record.nav_pose
            try:
                self._teleport_robot(target_pose)
            except Exception as exc:
                self.frontier_memory.fail(record.frontier_id, f"teleport failed: {exc}")
                self.last_error = f"Teleport failed for `{record.frontier_id}`: {exc}"
                self._log_decision("teleport_failed")
                self._prepare_decision_locked()
                return self.snapshot()

            reached_pose = self._current_pose()
            distance_m = _pose_distance_m(start_pose, reached_pose)
            self.total_distance_m += distance_m
            self.control_steps += 1
            self.trajectory.append(reached_pose.to_dict())
            self._perform_scan(
                full_turnaround=self._scan_uses_turnaround(),
                capture_frame=True,
                reason=self._scan_reason(f"teleport_arrive_frontier::{record.frontier_id}"),
            )
            self.frontier_memory.complete(record.frontier_id)
            record.path_cost_m = None
            record.currently_visible = False
            self.frontier_memory.remember_return_waypoint(
                room_id=None,
                pose=self._current_pose(),
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_decision("teleport_applied")
            self.pending_decision = None
            self.pending_trace = None
            self.applied_memory_updates = []
            self._prepare_decision_locked()
            return self.snapshot()

    def _initialize_environment(self) -> None:
        try:
            import gymnasium as gym
            from multido_xlerobot.maniskill import bootstrap_xlerobot_maniskill
        except Exception as exc:  # pragma: no cover - runtime dependency guard.
            raise RuntimeError(
                "The ManiSkill interactive backend requires gymnasium and the XLeRobot ManiSkill package "
                "in the active Python environment."
            ) from exc

        bootstrap_xlerobot_maniskill(self.options.repo_root, force_reload=self.options.force_reload)
        env_kwargs: dict[str, Any] = {
            "obs_mode": "sensor_data",
            "control_mode": self.options.control_mode,
            "render_mode": self.options.render_mode,
            "sensor_configs": {"shader_pack": self.options.shader},
            "human_render_camera_configs": {"shader_pack": self.options.shader},
            "viewer_camera_configs": {"shader_pack": self.options.shader},
            "robot_uids": self.options.robot_uid,
            "num_envs": self.options.num_envs,
            "sim_backend": self.options.sim_backend,
            "enable_shadow": True,
            "parallel_in_single_scene": False,
        }
        self.env = gym.make(self.options.env_id, **env_kwargs)

    def _reset_environment(self) -> None:
        reset_options: dict[str, Any] = {"reconfigure": True}
        if self.options.build_config_idx is not None:
            reset_options["build_config_idxs"] = self.options.build_config_idx
        self.env.reset(seed=2022, options=reset_options)
        self.robot = self.env.unwrapped.agent.robot
        self.base_link = self.env.unwrapped.agent.base_link
        self.head_link = self.env.unwrapped.agent.head_camera_link
        self.scene = self.env.unwrapped.scene
        self.action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self._base_qpos_anchor = np.asarray(self._tensor_to_numpy(self.robot.get_qpos()), dtype=np.float64).copy()
        self._base_world_anchor = self._current_pose()
        start_pose = _resolve_manishkill_start_pose(
            self._current_pose(),
            spawn_x=self.options.spawn_x,
            spawn_y=self.options.spawn_y,
            spawn_yaw=self.options.spawn_yaw,
            spawn_facing=self.options.spawn_facing,
        )
        if start_pose is not None:
            self._teleport_robot(start_pose)
            return
        self._settle_after_teleport()

    def _teleport_robot(self, pose: Pose2D) -> None:
        self._set_base_pose_via_qpos(pose)
        achieved = self._current_pose()
        position_error_m = math.hypot(achieved.x - pose.x, achieved.y - pose.y)
        yaw_error_rad = abs(_angle_wrap(achieved.yaw - pose.yaw))
        if position_error_m > BASE_TELEPORT_POSITION_TOLERANCE_M or yaw_error_rad > BASE_TELEPORT_YAW_TOLERANCE_RAD:
            raise RuntimeError(
                "Base teleport landed at "
                f"({achieved.x:.3f}, {achieved.y:.3f}, yaw={achieved.yaw:.3f}) instead of "
                f"({pose.x:.3f}, {pose.y:.3f}, yaw={pose.yaw:.3f}). "
                "This usually means the target pose is colliding with scene geometry or the base-frame/qpos conversion is still wrong."
            )

    def _set_base_pose_via_qpos(self, pose: Pose2D) -> None:
        updated_qpos = _updated_mobile_base_qpos(
            self.robot.get_qpos(),
            pose,
            anchor_pose=self._base_world_anchor,
            anchor_qpos=self._base_qpos_anchor,
        )
        self.robot.set_qpos(updated_qpos)
        try:
            self.robot.set_qvel(_zero_mobile_base_qvel(self.robot.get_qvel()))
        except Exception:
            pass
        self._settle_after_teleport()

    def _probe_teleport_pose(self, pose: Pose2D) -> tuple[bool, str | None]:
        saved_qpos = self.robot.get_qpos()
        saved_qvel = self.robot.get_qvel()
        try:
            self._teleport_robot(pose)
        except Exception as exc:
            try:
                self.robot.set_qpos(saved_qpos)
                self.robot.set_qvel(saved_qvel)
                self._settle_after_teleport()
            except Exception:
                pass
            return False, str(exc)
        try:
            self.robot.set_qpos(saved_qpos)
            self.robot.set_qvel(saved_qvel)
            self._settle_after_teleport()
        except Exception:
            pass
        return True, None

    def _resolve_physically_valid_target_pose(
        self,
        *,
        record: FrontierRecord,
        start_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
    ) -> tuple[Pose2D | None, str | None]:
        candidates = self._candidate_target_poses(
            record=record,
            start_pose=start_pose,
            reachable_safe_cells=reachable_safe_cells,
        )
        rejection_reasons: list[str] = []
        for candidate in candidates:
            ok, reason = self._probe_teleport_pose(candidate)
            if ok:
                return candidate, None
            if reason:
                rejection_reasons.append(reason)
        if rejection_reasons:
            return None, rejection_reasons[-1]
        return None, "no physically valid inward pose candidate was found"

    def _candidate_target_poses(
        self,
        *,
        record: FrontierRecord,
        start_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
    ) -> list[Pose2D]:
        boundary = record.centroid_pose
        nav = record.nav_pose
        inward_x = nav.x - boundary.x
        inward_y = nav.y - boundary.y
        inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x = nav.x - start_pose.x
            inward_y = nav.y - start_pose.y
            inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x, inward_y, inward_norm = 1.0, 0.0, 1.0
        inward_x /= inward_norm
        inward_y /= inward_norm

        offsets_m = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
        resolution = self.config.occupancy_resolution
        search_radius_cells = max(1, int(math.ceil(0.50 / resolution)))
        candidates: list[Pose2D] = []
        seen_cells: set[GridCell] = set()
        for offset_m in offsets_m:
            desired_x = nav.x + inward_x * offset_m
            desired_y = nav.y + inward_y * offset_m
            desired_cell = self._world_to_cell(desired_x, desired_y)
            snapped_cell: GridCell | None = None
            snapped_score: float | None = None
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    cell = GridCell(desired_cell.x + dx, desired_cell.y + dy)
                    if cell not in reachable_safe_cells or cell in seen_cells:
                        continue
                    cell_pose = cell.center_pose(resolution)
                    score = (
                        math.hypot(cell_pose.x - desired_x, cell_pose.y - desired_y)
                        + 0.02 * _pose_distance_m(cell_pose, start_pose)
                    )
                    if snapped_score is None or score < snapped_score:
                        snapped_cell = cell
                        snapped_score = score
            if snapped_cell is None:
                continue
            seen_cells.add(snapped_cell)
            snapped_pose = snapped_cell.center_pose(
                resolution,
                yaw=math.atan2(boundary.y - snapped_cell.center_pose(resolution).y, boundary.x - snapped_cell.center_pose(resolution).x),
            )
            candidates.append(snapped_pose)
        return candidates

    def _settle_after_teleport(self) -> None:
        steps = max(int(self.options.teleport_settle_steps), 0)
        for _ in range(steps):
            self.env.step(self.action)
        if self.options.render_mode == "human":
            self.env.render()

    def _tensor_to_numpy(self, value: Any) -> np.ndarray:
        array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        return np.asarray(array, dtype=np.float64).squeeze()

    def _current_pose(self) -> Pose2D:
        position = self._tensor_to_numpy(self.base_link.pose.p)
        quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.base_link.pose.q))
        return Pose2D(float(position[0]), float(position[1]), quaternion_to_yaw(quaternion))

    def _display_pose(self, pose: Pose2D) -> Pose2D:
        return Pose2D(pose.x, pose.y, _angle_wrap(pose.yaw + self._display_yaw_offset_rad))

    def _scan_uses_turnaround(self) -> bool:
        return self.options.scan_mode == "turnaround"

    def _scan_reason(self, stage: str) -> str:
        if self._scan_uses_turnaround():
            return f"maniskill_{stage}_turnaround_scan"
        return f"maniskill_{stage}_forward_scan"

    def _world_to_cell(self, x: float, y: float) -> GridCell:
        resolution = self.config.occupancy_resolution
        return GridCell(int(math.floor(x / resolution)), int(math.floor(y / resolution)))

    def _capture_head_camera(self) -> dict[str, np.ndarray]:
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        camera = self.scene.sensors[HEAD_CAMERA_UID]
        camera.capture()
        data = camera.get_obs(rgb=True, depth=True, position=False, segmentation=False)
        converted: dict[str, np.ndarray] = {}
        for key, value in data.items():
            array = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
            converted[key] = np.asarray(array).squeeze()
        return converted

    def _perform_scan(self, *, full_turnaround: bool, capture_frame: bool, reason: str) -> None:
        original_pose = self._current_pose()
        sample_count = max(int(self.options.scan_yaw_samples), 1) if full_turnaround else 1
        yaws = [original_pose.yaw + (math.tau * index / sample_count) for index in range(sample_count)]
        capture_indices = {0}
        if full_turnaround:
            capture_indices = {min(sample_count - 1, round(sample_count * fraction)) for fraction in (0.0, 0.25, 0.5, 0.75)}
        captured_any = False
        previous_active_scan_cells = getattr(self, "_active_scan_cells", None)
        previous_active_scan_known_cells = getattr(self, "_active_scan_known_cells", None)
        previous_active_scan_range_edge_cells = getattr(self, "_active_scan_range_edge_cells", None)
        self._active_scan_cells: set[GridCell] | None = set()
        self._active_scan_known_cells: dict[GridCell, str] | None = {}
        self._active_scan_range_edge_cells: set[GridCell] | None = set()
        try:
            for index, yaw in enumerate(yaws):
                self._teleport_robot(Pose2D(original_pose.x, original_pose.y, yaw))
                head_data = self._capture_head_camera()
                summary = self._integrate_depth_scan(head_data)
                if capture_frame and index in capture_indices:
                    self._append_keyframe(
                        head_data=head_data,
                        summary=summary,
                        reason=f"{reason}::yaw_sample_{index:02d}",
                    )
                    captured_any = True
            if full_turnaround:
                self._teleport_robot(original_pose)
            if capture_frame and not captured_any:
                head_data = self._capture_head_camera()
                self._append_keyframe(
                    head_data=head_data,
                    summary=self._integrate_depth_scan(head_data),
                    reason=reason,
                )
        finally:
            self.latest_scan_known_cells = dict(self._active_scan_known_cells or {})
            self.latest_scan_range_edge_cells = set(self._active_scan_range_edge_cells or set())
            self.latest_scan_cells = set(self.latest_scan_known_cells)
            self._merge_latest_scan_into_global()
            self._active_scan_cells = previous_active_scan_cells
            self._active_scan_known_cells = previous_active_scan_known_cells
            self._active_scan_range_edge_cells = previous_active_scan_range_edge_cells

    def _integrate_depth_scan(self, head_data: dict[str, np.ndarray]) -> dict[str, Any]:
        depth = np.asarray(head_data.get("depth"))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_for_scan = _depth_image_to_millimeters(depth)
        ranges, angles = synthesize_scan_from_depth(
            depth_for_scan,
            horizontal_fov_rad=HEAD_CAMERA_FOV_RAD,
            band_height_px=12,
            range_min_m=0.05,
            range_max_m=self.config.sensor_range_m,
        )
        head_position = self._tensor_to_numpy(self.head_link.pose.p)
        head_quaternion = normalize_quaternion_wxyz(self._tensor_to_numpy(self.head_link.pose.q))
        laser_yaw = quaternion_to_yaw(head_quaternion)
        origin_cell = self._world_to_cell(float(head_position[0]), float(head_position[1]))
        active_scan_cells = getattr(self, "_active_scan_cells", None)
        active_scan_known_cells = getattr(self, "_active_scan_known_cells", None)
        active_scan_range_edge_cells = getattr(self, "_active_scan_range_edge_cells", None)
        if active_scan_cells is not None:
            active_scan_cells.add(origin_cell)
        if active_scan_known_cells is not None:
            active_scan_known_cells[origin_cell] = "free"

        resolution = self.config.occupancy_resolution
        step_m = max(resolution * 0.5, 0.05)
        beam_stride = max(int(self.options.depth_beam_stride), 1)
        point_count = 0
        for index in range(0, len(ranges), beam_stride):
            beam_range = float(ranges[index])
            hit_obstacle = math.isfinite(beam_range) and beam_range < self.config.sensor_range_m * 0.98
            ray_max_m = min(beam_range, self.config.sensor_range_m) if math.isfinite(beam_range) else self.config.sensor_range_m
            if ray_max_m <= 0.05:
                continue
            angle = laser_yaw + float(angles[index])
            last_free: GridCell | None = None
            samples = max(1, int(ray_max_m / step_m))
            for sample_index in range(1, samples + 1):
                distance_m = min(sample_index * step_m, ray_max_m)
                cell = self._world_to_cell(
                    float(head_position[0]) + math.cos(angle) * distance_m,
                    float(head_position[1]) + math.sin(angle) * distance_m,
                )
                if active_scan_cells is not None:
                    active_scan_cells.add(cell)
                if hit_obstacle and sample_index == samples and cell != origin_cell:
                    if active_scan_known_cells is not None:
                        active_scan_known_cells[cell] = "occupied"
                    point_count += 1
                else:
                    if active_scan_known_cells is not None:
                        active_scan_known_cells[cell] = "free"
                    last_free = cell
                    point_count += 1
            if not hit_obstacle and last_free is not None:
                if active_scan_range_edge_cells is not None:
                    active_scan_range_edge_cells.add(last_free)

        summary = _depth_summary(depth_for_scan, max_range_m=self.config.sensor_range_m)
        summary.update(
            {
                "point_count": point_count,
                "scan_beams": int(len(ranges)),
                "integrated_beams": int(math.ceil(len(ranges) / beam_stride)),
                "source": "maniskill_xlerobot_head_rgbd",
            }
        )
        return summary

    def _merge_latest_scan_into_global(self) -> None:
        for cell, state in self.latest_scan_known_cells.items():
            self.known_cells[cell] = state
        self.range_edge_cells |= self.latest_scan_range_edge_cells

    def _append_keyframe(self, *, head_data: dict[str, np.ndarray], summary: dict[str, Any], reason: str) -> None:
        pose = self._current_pose()
        frame_id = f"kf_{len(self.keyframes) + 1:03d}"
        frame = {
            "frame_id": frame_id,
            "pose": pose.to_dict(),
            "region_id": "unknown",
            "visible_objects": [],
            "point_count": int(summary.get("point_count", 0)),
            "depth_min_m": summary.get("depth_min_m"),
            "depth_max_m": summary.get("depth_max_m"),
            "description": (
                f"Real ManiSkill XLeRobot head RGB-D frame captured for `{reason}` at "
                f"world pose ({pose.x:.2f}, {pose.y:.2f}, yaw={pose.yaw:.2f})."
            ),
            "rgbd_summary": summary,
            "thumbnail_data_url": _rgb_array_to_data_url(head_data.get("rgb")),
        }
        self.keyframes.append(frame)

    def _prepare_decision_locked(self) -> None:
        self.decision_index += 1
        visible_candidates = self._detect_frontier_candidates()
        self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
        candidate_records = self._refresh_candidate_paths()
        prompt_payload = self._build_prompt_payload(candidate_records)
        self.pending_candidate_records = candidate_records
        self.pending_prompt_payload = prompt_payload
        self.pending_prompt_text = build_exploration_policy_user_prompt(prompt_payload)
        self.pending_decision = None
        self.pending_trace = None
        self.applied_memory_updates = []
        self.status = "waiting_for_llm" if candidate_records else "finished"

    def _known_free_cells(self) -> set[GridCell]:
        return {cell for cell, state in self.known_cells.items() if state == "free"}

    def _global_frontier_components(
        self,
        cluster: list[GridCell],
        *,
        scan_range_edge_cells: set[GridCell],
    ) -> list[tuple[list[GridCell], set[GridCell], int]]:
        valid_frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        for cell in cluster:
            if self.known_cells.get(cell) != "free":
                continue
            unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in self.known_cells}
            if unknown_neighbors:
                valid_frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        components: list[tuple[list[GridCell], set[GridCell], int]] = []
        visited: set[GridCell] = set()
        for cell in valid_frontier_cells:
            if cell in visited:
                continue
            component: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in valid_frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component_unknown = set().union(
                *(unknown_neighbors_by_frontier.get(component_cell, set()) for component_cell in component)
            )
            if component_unknown:
                max_unknown_neighbor_count = max(
                    len(unknown_neighbors_by_frontier.get(component_cell, set()))
                    for component_cell in component
                )
                components.append((component, component_unknown, max_unknown_neighbor_count))
        return components

    def _global_frontier_anchor_cell_near_record(
        self,
        record: FrontierRecord,
    ) -> tuple[GridCell | None, str | None]:
        known_cells = getattr(self, "known_cells", {})
        range_edge_cells = getattr(self, "range_edge_cells", set())
        resolution = self.config.occupancy_resolution
        boundary_cell = self._world_to_cell(record.centroid_pose.x, record.centroid_pose.y)
        search_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVALIDATION_RADIUS_M / resolution)))
        strong_candidates: list[tuple[int, int, GridCell]] = []
        relaxed_candidates: list[tuple[int, int, GridCell]] = []
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = GridCell(boundary_cell.x + dx, boundary_cell.y + dy)
                distance_cells = _grid_distance_cells(cell, boundary_cell)
                if distance_cells > search_radius_cells:
                    continue
                if known_cells.get(cell) != "free":
                    continue
                unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in known_cells}
                if len(unknown_neighbors) >= 2 or (cell in range_edge_cells and unknown_neighbors):
                    strong_candidates.append((distance_cells, -len(unknown_neighbors), cell))
                elif unknown_neighbors:
                    relaxed_candidates.append((distance_cells, -len(unknown_neighbors), cell))
        if strong_candidates:
            return min(strong_candidates, key=lambda item: (item[0], item[1]))[2], "strong"
        if relaxed_candidates:
            return min(relaxed_candidates, key=lambda item: (item[0], item[1]))[2], "relaxed"
        return None, None

    def _record_still_has_global_frontier_boundary(self, record: FrontierRecord) -> bool:
        anchor_cell, _anchor_mode = self._global_frontier_anchor_cell_near_record(record)
        return anchor_cell is not None

    def _revalidate_stored_frontier_boundary(
        self,
        *,
        record: FrontierRecord,
        anchor_cell: GridCell,
        anchor_mode: str | None,
    ) -> None:
        anchor_pose = anchor_cell.center_pose(self.config.occupancy_resolution)
        relaxed_note = (
            "stored frontier memory was kept using relaxed revalidation because nearby free space still borders "
            "unknown map area even though the stricter current-frontier rule no longer matches the original point"
            if anchor_mode == "relaxed"
            else None
        )
        if (
            _pose_distance_m(record.centroid_pose, anchor_pose) <= self.config.occupancy_resolution * 0.5
            and not relaxed_note
        ):
            return
        previous_pose = record.centroid_pose
        record.centroid_pose = anchor_pose
        record.evidence = _dedupe_text(
            record.evidence
            + [
                *(
                    [
                        (
                            "stored frontier boundary was revalidated against the current global occupancy map "
                            f"near the original memory point ({previous_pose.x:.2f}, {previous_pose.y:.2f})"
                        )
                    ]
                    if _pose_distance_m(previous_pose, anchor_pose) > self.config.occupancy_resolution * 0.5
                    else []
                ),
                *( [relaxed_note] if relaxed_note else [] ),
            ]
        )

    def _resnap_stored_frontier_revisit_pose(
        self,
        *,
        record: FrontierRecord,
        current_pose: Pose2D,
        reachable_safe_cells: set[GridCell],
        anchor_cell: GridCell,
    ) -> Pose2D | None:
        candidates = self._candidate_target_poses(
            record=record,
            start_pose=current_pose,
            reachable_safe_cells=reachable_safe_cells,
        )
        if not candidates:
            resolution = self.config.occupancy_resolution
            target_distance_m = self._robot_footprint_radius_m() + 0.25
            max_radius_cells = max(1, int(math.ceil(STORED_FRONTIER_REVISIT_APPROACH_RADIUS_M / resolution)))
            anchor_pose = anchor_cell.center_pose(resolution)
            scored_cells: list[tuple[float, GridCell]] = []
            for cell in reachable_safe_cells:
                distance_cells = _grid_distance_cells(cell, anchor_cell)
                if distance_cells > max_radius_cells:
                    continue
                cell_pose = cell.center_pose(resolution)
                score = (
                    abs((distance_cells * resolution) - target_distance_m)
                    + 0.03 * _pose_distance_m(cell_pose, current_pose)
                    + 0.02 * _pose_distance_m(cell_pose, record.nav_pose)
                )
                scored_cells.append((score, cell))
            for _score, cell in sorted(scored_cells, key=lambda item: item[0]):
                cell_pose = cell.center_pose(resolution)
                candidates.append(
                    cell.center_pose(
                        resolution,
                        yaw=math.atan2(anchor_pose.y - cell_pose.y, anchor_pose.x - cell_pose.x),
                    )
                )
                break
        if not candidates:
            return None
        return min(candidates, key=lambda pose: _pose_distance_m(current_pose, pose))

    def _apply_stored_frontier_resnap(
        self,
        *,
        record: FrontierRecord,
        target_pose: Pose2D,
        previous_pose: Pose2D,
    ) -> None:
        if _pose_distance_m(target_pose, previous_pose) <= self.config.occupancy_resolution * 0.5:
            return
        record.nav_pose = target_pose
        record.evidence = _dedupe_text(
            record.evidence
            + [
                (
                    "stored frontier revisit approach pose was re-snapped to robot-connected, "
                    "footprint-eroded free space before sending the frontier back to the LLM"
                )
            ]
        )
        self.guardrail_events.append(
            {
                "type": "stored_frontier_revisit_pose_resnapped",
                "frontier_id": record.frontier_id,
                "previous_nav_pose": previous_pose.to_dict(),
                "resnapped_nav_pose": target_pose.to_dict(),
            }
        )

    def _detect_frontier_candidates(self) -> list[FrontierCandidate]:
        robot_pose = self._current_pose()
        robot_cell = self._world_to_cell(robot_pose.x, robot_pose.y)
        reachable_safe_cells = self._reachable_safe_navigation_cells(robot_cell)
        scan_known_cells = dict(getattr(self, "latest_scan_known_cells", {}))
        scan_range_edge_cells = set(getattr(self, "latest_scan_range_edge_cells", set()))
        if not scan_known_cells:
            scan_known_cells = dict(self.known_cells)
            scan_range_edge_cells = set(self.range_edge_cells)
        candidate_source_cells = {
            cell
            for cell, state in scan_known_cells.items()
            if state == "free"
        }
        frontier_cells: set[GridCell] = set()
        unknown_neighbors_by_frontier: dict[GridCell, set[GridCell]] = {}
        for cell in candidate_source_cells:
            unknown_neighbors = {neighbor for neighbor in _neighbors4(cell) if neighbor not in scan_known_cells}
            if unknown_neighbors:
                frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        candidates: list[FrontierCandidate] = []
        for local_cluster in clusters:
            local_cluster_unknown = set().union(
                *(unknown_neighbors_by_frontier.get(cell, set()) for cell in local_cluster)
            )
            if not local_cluster_unknown:
                continue
            global_components = self._global_frontier_components(
                local_cluster,
                scan_range_edge_cells=scan_range_edge_cells,
            )
            if not global_components:
                self.guardrail_events.append(
                    {
                        "type": "frontier_not_global_boundary_after_merge",
                        "local_cluster_size": len(local_cluster),
                        "local_unknown_gain": len(local_cluster_unknown),
                        "frontier_boundary_pose": _cell_mean_pose(local_cluster, self.config.occupancy_resolution).to_dict(),
                    }
                )
                continue

            for cluster, cluster_unknown, max_unknown_neighbor_count in global_components:
                approach_cell = self._select_frontier_approach_cell(
                    cluster=cluster,
                    unknown_cells=cluster_unknown,
                    robot_cell=robot_cell,
                    reachable_safe_cells=reachable_safe_cells,
                )
                if approach_cell is None:
                    self.guardrail_events.append(
                        {
                            "type": "frontier_without_safe_approach_pose",
                            "cluster_size": len(cluster),
                            "unknown_gain": len(cluster_unknown),
                            "frontier_boundary_pose": _cell_mean_pose(cluster, self.config.occupancy_resolution).to_dict(),
                            "robot_footprint_radius_m": round(self._robot_footprint_radius_m(), 3),
                        }
                    )
                    continue
                centroid_cell = _centroid_cell(cluster)
                boundary_pose = _cell_mean_pose(cluster, self.config.occupancy_resolution)
                nav_yaw = math.atan2(boundary_pose.y - approach_cell.center_pose(self.config.occupancy_resolution).y, boundary_pose.x - approach_cell.center_pose(self.config.occupancy_resolution).x)
                nav_pose = approach_cell.center_pose(self.config.occupancy_resolution, yaw=nav_yaw)
                centroid_pose = centroid_cell.center_pose(self.config.occupancy_resolution)
                evidence = [
                    f"{len(cluster_unknown)} unknown neighbor cells still unknown after merging this RGB-D scan into the global occupancy map",
                    f"cluster size {len(cluster)}",
                    f"frontier boundary centroid is at ({boundary_pose.x:.2f}, {boundary_pose.y:.2f})",
                    "frontier passed merged-map validation, so local scan edges buried inside known free space are rejected",
                    (
                        "approach pose is offset inward and drawn from robot-connected, footprint-eroded free space "
                        f"for a {self._robot_footprint_radius_m():.2f} m XLeRobot/IKEA-cart footprint radius"
                    ),
                    "frontier information from actual XLeRobot head RGB-D, not complete apartment knowledge",
                    "movement in this playground is teleport-only and does not prove Nav2 reachability",
                ]
                if max_unknown_neighbor_count >= 2:
                    evidence.append(
                        "frontier signal is stronger: multiple unknown-facing neighbors support likely map expansion"
                    )
                else:
                    evidence.append(
                        "frontier signal is weaker: a single unknown-facing edge was kept because robot-sized free space exists nearby; the LLM should veto it if RGB/map evidence suggests no useful opening"
                    )
                if any(cell in scan_range_edge_cells for cell in cluster):
                    evidence.append("frontier aligns with an RGB-D sensor range limit")
                candidates.append(
                    FrontierCandidate(
                        frontier_id=None,
                        member_cells=tuple(sorted(cluster)),
                        nav_cell=approach_cell,
                        centroid_cell=centroid_cell,
                        nav_pose=nav_pose,
                        centroid_pose=centroid_pose,
                        unknown_gain=len(cluster_unknown),
                        sensor_range_edge=any(cell in scan_range_edge_cells for cell in cluster),
                        room_hint=None,
                        evidence=evidence,
                        currently_visible=_pose_distance_m(nav_pose, robot_pose) <= self.config.sensor_range_m + 0.5,
                    )
                )
        candidates.sort(
            key=lambda candidate: (
                _grid_distance_cells(candidate.nav_cell, robot_cell),
                -candidate.unknown_gain,
            )
        )
        return candidates[: max(int(self.options.max_frontiers), 1)]

    def _select_frontier_approach_cell(
        self,
        *,
        cluster: list[GridCell],
        unknown_cells: set[GridCell],
        robot_cell: GridCell,
        reachable_safe_cells: set[GridCell],
    ) -> GridCell | None:
        resolution = self.config.occupancy_resolution
        known_free = reachable_safe_cells
        if not cluster or not unknown_cells:
            return None

        frontier_x = sum((cell.x + 0.5) * resolution for cell in cluster) / len(cluster)
        frontier_y = sum((cell.y + 0.5) * resolution for cell in cluster) / len(cluster)
        unknown_x = sum((cell.x + 0.5) * resolution for cell in unknown_cells) / len(unknown_cells)
        unknown_y = sum((cell.y + 0.5) * resolution for cell in unknown_cells) / len(unknown_cells)
        inward_x = frontier_x - unknown_x
        inward_y = frontier_y - unknown_y
        inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            robot_pose = robot_cell.center_pose(resolution)
            inward_x = robot_pose.x - frontier_x
            inward_y = robot_pose.y - frontier_y
            inward_norm = math.hypot(inward_x, inward_y)
        if inward_norm <= 1e-6:
            inward_x, inward_y, inward_norm = 1.0, 0.0, 1.0
        inward_x /= inward_norm
        inward_y /= inward_norm

        footprint_radius_m = self._robot_footprint_radius_m()
        offsets_m = [
            footprint_radius_m + 0.05,
            footprint_radius_m + 0.20,
            footprint_radius_m + 0.35,
            footprint_radius_m + 0.55,
            footprint_radius_m + 0.80,
        ]
        projected_candidates: list[tuple[float, GridCell]] = []
        for offset_m in offsets_m:
            desired_x = frontier_x + inward_x * offset_m
            desired_y = frontier_y + inward_y * offset_m
            desired_cell = self._world_to_cell(desired_x, desired_y)
            search_radius_cells = max(1, int(math.ceil(0.40 / resolution)))
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    cell = GridCell(desired_cell.x + dx, desired_cell.y + dy)
                    if cell not in known_free:
                        continue
                    cell_pose = cell.center_pose(resolution)
                    score = (
                        math.hypot(cell_pose.x - desired_x, cell_pose.y - desired_y)
                        + 0.04 * _grid_distance_cells(cell, robot_cell) * resolution
                        + 0.10 / max(_grid_distance_cells(cell, _centroid_cell(cluster)), 1)
                    )
                    projected_candidates.append((score, cell))

        if not projected_candidates:
            fallback_radius_cells = max(1, int(math.ceil((footprint_radius_m + 0.65) / resolution)))
            boundary = _centroid_cell(cluster)
            for cell in known_free:
                distance_cells = _grid_distance_cells(cell, boundary)
                if distance_cells > fallback_radius_cells:
                    continue
                score = (
                    abs((distance_cells * resolution) - (footprint_radius_m + 0.25))
                    + 0.04 * _grid_distance_cells(cell, robot_cell) * resolution
                )
                projected_candidates.append((score, cell))

        seen: set[GridCell] = set()
        for _score, cell in sorted(projected_candidates, key=lambda item: item[0]):
            if cell in seen:
                continue
            seen.add(cell)
            if self._is_valid_robot_center_cell(
                cell,
                required_known_fraction=STRICT_NAVIGATION_KNOWN_FRACTION,
                unknown_is_blocking=True,
                extra_clearance_m=STRICT_NAVIGATION_CLEARANCE_MARGIN_M,
            ):
                return cell
        return None

    def _reachable_safe_navigation_cells(self, robot_cell: GridCell) -> set[GridCell]:
        safe_cells = {
            cell
            for cell in self._known_free_cells()
            if self._is_valid_robot_center_cell(
                cell,
                required_known_fraction=STRICT_NAVIGATION_KNOWN_FRACTION,
                unknown_is_blocking=True,
                extra_clearance_m=STRICT_NAVIGATION_CLEARANCE_MARGIN_M,
            )
        }
        if not safe_cells:
            return set()
        if robot_cell in safe_cells:
            seed = robot_cell
        else:
            seed = min(safe_cells, key=lambda cell: _grid_distance_cells(cell, robot_cell), default=None)
        if seed is None:
            return set()
        reachable: set[GridCell] = set()
        queue = [seed]
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in _neighbors4(current):
                if neighbor in safe_cells and neighbor not in reachable:
                    queue.append(neighbor)
        return reachable

    def _is_valid_robot_center_cell(
        self,
        cell: GridCell,
        *,
        required_known_fraction: float = 0.55,
        unknown_is_blocking: bool = False,
        extra_clearance_m: float = 0.0,
    ) -> bool:
        if self.known_cells.get(cell) != "free":
            return False
        resolution = self.config.occupancy_resolution
        clearance_radius_m = self._robot_footprint_radius_m() + max(float(extra_clearance_m), 0.0)
        radius_cells = max(1, int(math.ceil(clearance_radius_m / resolution)))
        known_count = 0
        footprint_count = 0
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                sample = GridCell(cell.x + dx, cell.y + dy)
                distance_m = math.hypot(dx * resolution, dy * resolution)
                if distance_m > clearance_radius_m:
                    continue
                footprint_count += 1
                state = self.known_cells.get(sample)
                if state == "occupied":
                    return False
                if state is None and unknown_is_blocking:
                    return False
                if state == "free":
                    known_count += 1
        if footprint_count == 0:
            return False
        return known_count / footprint_count >= required_known_fraction

    def _robot_footprint_radius_m(self) -> float:
        return max(
            float(self.config.robot_radius_m),
            XLEROBOT_IKEA_CART_FOOTPRINT_RADIUS_M + XLEROBOT_IKEA_CART_CLEARANCE_PADDING_M,
        )

    def _refresh_candidate_paths(self) -> list[FrontierRecord]:
        current_pose = self._current_pose()
        current_cell = self._world_to_cell(current_pose.x, current_pose.y)
        reachable_safe_cells = self._reachable_safe_navigation_cells(current_cell)
        current_pose_filter_m = max(self.config.occupancy_resolution * 1.5, self._robot_footprint_radius_m())
        reachable_records: list[FrontierRecord] = []
        for record in self.frontier_memory.candidate_records():
            if record.status == "active" or record.frontier_id == self.frontier_memory.active_frontier_id:
                record.path_cost_m = None
                self.guardrail_events.append(
                    {
                        "type": "active_frontier_filtered_from_prompt",
                        "frontier_id": record.frontier_id,
                    }
                )
                continue
            frontier_anchor_cell: GridCell | None = None
            frontier_anchor_mode: str | None = None
            if not record.currently_visible:
                frontier_anchor_cell, frontier_anchor_mode = self._global_frontier_anchor_cell_near_record(record)
                if frontier_anchor_cell is None:
                    record.path_cost_m = None
                    self.guardrail_events.append(
                        {
                            "type": "stored_frontier_not_global_boundary_after_merge",
                            "frontier_id": record.frontier_id,
                            "frontier_boundary_pose": record.centroid_pose.to_dict(),
                            "search_radius_m": STORED_FRONTIER_REVALIDATION_RADIUS_M,
                        }
                    )
                    continue
                self._revalidate_stored_frontier_boundary(
                    record=record,
                    anchor_cell=frontier_anchor_cell,
                    anchor_mode=frontier_anchor_mode,
                )
            target_cell = self._world_to_cell(record.nav_pose.x, record.nav_pose.y)
            path_cost_m = _pose_distance_m(current_pose, record.nav_pose) if target_cell in reachable_safe_cells else None
            if path_cost_m is None and not record.currently_visible and frontier_anchor_cell is not None:
                previous_pose = record.nav_pose
                resnapped_pose = self._resnap_stored_frontier_revisit_pose(
                    record=record,
                    current_pose=current_pose,
                    reachable_safe_cells=reachable_safe_cells,
                    anchor_cell=frontier_anchor_cell,
                )
                if resnapped_pose is not None:
                    self._apply_stored_frontier_resnap(
                        record=record,
                        target_pose=resnapped_pose,
                        previous_pose=previous_pose,
                    )
                    target_cell = self._world_to_cell(record.nav_pose.x, record.nav_pose.y)
                    path_cost_m = (
                        _pose_distance_m(current_pose, record.nav_pose)
                        if target_cell in reachable_safe_cells
                        else None
                    )
                if path_cost_m is None:
                    record.path_cost_m = None
                    self.guardrail_events.append(
                        {
                            "type": "stored_frontier_without_reachable_revisit_pose",
                            "frontier_id": record.frontier_id,
                            "frontier_boundary_pose": record.centroid_pose.to_dict(),
                        }
                    )
                    continue
            record.path_cost_m = path_cost_m
            if record.path_cost_m is None:
                self.guardrail_events.append(
                    {
                        "type": "frontier_without_reachable_nav_pose",
                        "frontier_id": record.frontier_id,
                        "currently_visible": record.currently_visible,
                    }
                )
                continue
            if target_cell == current_cell or record.path_cost_m <= current_pose_filter_m:
                record.path_cost_m = None
                self.guardrail_events.append(
                    {
                        "type": "frontier_at_current_pose_filtered",
                        "frontier_id": record.frontier_id,
                        "path_cost_m": round(path_cost_m, 3),
                        "filter_radius_m": round(current_pose_filter_m, 3),
                    }
                )
                continue
            reachable_records.append(record)

        updated = reachable_records
        updated.sort(
            key=lambda record: (
                record.path_cost_m if record.path_cost_m is not None else 1e9,
                not record.currently_visible,
                -(record.llm_memory_priority or 0.0),
                -record.unknown_gain,
            )
        )
        return updated[: max(int(self.options.max_frontiers), 1)]

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        known_free = sum(1 for state in self.known_cells.values() if state == "free")
        occupied = sum(1 for state in self.known_cells.values() if state == "occupied")
        navigation_map_image = _navigation_map_data_url(
            known_cells=self.known_cells,
            resolution=self.config.occupancy_resolution,
            trajectory=self.trajectory,
            robot_pose=self._current_pose(),
            candidate_records=candidate_records,
            remembered_records=[],
        )
        return {
            "mission": (
                "Interactive ManiSkill no-Nav2 test: use actual XLeRobot head RGB-D images and a deterministic "
                "2D RGB-D occupancy projection to choose useful robot-navigable exploration regions. "
                "The movement executor will teleport the robot after operator review, so this tests LLM frontier choice "
                "and visual interpretation, not Nav2 control."
            ),
            "robot": {
                "pose": self._current_pose().to_dict(),
                "room_id": None,
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
                "navigation_mode": "maniskill_teleport_no_nav2",
                "rgbd_source": "actual_manishkill_xlerobot_head_camera",
            },
            "frontier_memory": _frontier_memory_prompt_context(self.frontier_memory),
            "frontier_information": [record.to_dict() for record in candidate_records],
            "frontier_selection_guidance": _frontier_selection_guidance(),
            "explored_areas": [
                {
                    "region_id": "maniskill_mapped_space",
                    "label": "mapped_space",
                    "observed_fraction": round(self._coverage(), 3),
                    "objects_seen": [],
                    "representative_frames": [item["frame_id"] for item in self.keyframes[-6:]],
                }
            ],
            "navigation_map_views": [
                {
                    "frame_id": f"nav_map_{self.decision_index:03d}",
                    "description": (
                        "Operator-review navigation map rendered from the same occupancy/frontier data as the web UI. "
                        "Green frontier labels correspond exactly to Frontier Information; red is the robot and heading."
                    ),
                    "thumbnail_data_url": navigation_map_image,
                    "frontier_count": len(candidate_records),
                }
            ] if navigation_map_image else [],
            "recent_views": self.keyframes[-6:],
            "map_stats": {
                "known_free_cells": known_free,
                "occupied_cells": occupied,
                "known_cells": len(self.known_cells),
                "map_bounds": self._map_bounds(),
                "robot_footprint": {
                    "source": "xlerobot URDF base_link IKEA/RASKOG cart collision boxes",
                    "length_m": XLEROBOT_IKEA_CART_FOOTPRINT_LENGTH_M,
                    "width_m": XLEROBOT_IKEA_CART_FOOTPRINT_WIDTH_M,
                    "clearance_radius_m": round(self._robot_footprint_radius_m(), 3),
                },
            },
            "guardrails": {
                "frontier_information_is_partial_rgbd_scan_evidence": True,
                "uses_actual_manishkill_rgbd": True,
                "navigation_is_teleport_mocked_without_nav2": True,
                "frontier_ids_must_come_from_prompt": True,
                "memory_updates_must_use_existing_frontier_ids": True,
                "avoid_furniture_shadow_boundaries_without_clear_open_space": True,
                "do_not_claim_nav2_reachability_from_this_playground": True,
            },
            "ascii_map": self._ascii_map(candidate_records),
        }

    def _ascii_map(self, candidate_records: list[FrontierRecord]) -> str:
        if not self.known_cells:
            return "map unavailable"
        robot_pose = self._current_pose()
        robot_cell = self._world_to_cell(robot_pose.x, robot_pose.y)
        frontier_cells = {
            self._world_to_cell(record.nav_pose.x, record.nav_pose.y): record.status
            for record in candidate_records
        }
        interesting = set(self.known_cells) | set(frontier_cells) | {robot_cell}
        min_x = min(cell.x for cell in interesting) - 2
        max_x = max(cell.x for cell in interesting) + 2
        min_y = min(cell.y for cell in interesting) - 2
        max_y = max(cell.y for cell in interesting) + 2
        max_width = 72
        max_height = 48
        if max_x - min_x + 1 > max_width:
            half = max_width // 2
            min_x = robot_cell.x - half
            max_x = min_x + max_width - 1
        if max_y - min_y + 1 > max_height:
            half = max_height // 2
            min_y = robot_cell.y - half
            max_y = min_y + max_height - 1
        lines: list[str] = []
        for y in reversed(range(min_y, max_y + 1)):
            row: list[str] = []
            for x in range(min_x, max_x + 1):
                cell = GridCell(x, y)
                if cell == robot_cell:
                    row.append("R")
                elif cell in frontier_cells:
                    row.append("V" if frontier_cells[cell] in {"completed", "failed", "suppressed"} else "F")
                else:
                    state = self.known_cells.get(cell)
                    row.append("." if state == "free" else "#" if state == "occupied" else "?")
            lines.append("".join(row))
        return "\n".join(lines)

    def _apply_finish_guardrail(
        self,
        decision: ExplorationDecision,
        candidate_records: list[FrontierRecord],
    ) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable:
            fallback = self.policy._heuristic_decision(
                candidate_records,
                list(self.frontier_memory.return_waypoints.values()),
                self._coverage(),
                None,
            )
            self.guardrail_events.append(
                {
                    "type": "finish_with_reachable_frontiers",
                    "requested": decision.to_dict(),
                    "fallback": fallback.to_dict(),
                    "reachable_frontier_ids": [record.frontier_id for record in reachable],
                }
            )
            return fallback
        return decision

    def _pending_target(self) -> dict[str, Any] | None:
        if self.pending_decision is None or not self.pending_decision.selected_frontier_id:
            return None
        record = self.frontier_memory.records.get(self.pending_decision.selected_frontier_id)
        if record is None:
            return None
        return {
            "frontier_id": record.frontier_id,
            "nav_pose": record.nav_pose.to_dict(),
            "centroid_pose": record.centroid_pose.to_dict(),
            "status": record.status,
            "path_cost_m": record.path_cost_m,
        }

    def _log_decision(self, event_type: str) -> None:
        if self.pending_decision is None:
            return
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "event_type": event_type,
                "coverage": round(self._coverage(), 3),
                "decision": self.pending_decision.to_dict(),
                "trace": self.pending_trace or {},
                "applied_memory_updates": list(self.applied_memory_updates),
            }
        )

    def _coverage(self) -> float:
        if not self.known_cells:
            return 0.0
        min_x = min(cell.x for cell in self.known_cells) - 3
        max_x = max(cell.x for cell in self.known_cells) + 3
        min_y = min(cell.y for cell in self.known_cells) - 3
        max_y = max(cell.y for cell in self.known_cells) + 3
        bbox_cells = max((max_x - min_x + 1) * (max_y - min_y + 1), 1)
        return min(len(self.known_cells) / bbox_cells, 1.0)

    def _map_bounds(self) -> dict[str, float]:
        resolution = self.config.occupancy_resolution
        if not self.known_cells:
            pose = self._current_pose()
            return {
                "min_x": round(pose.x - 1.0, 3),
                "max_x": round(pose.x + 1.0, 3),
                "min_y": round(pose.y - 1.0, 3),
                "max_y": round(pose.y + 1.0, 3),
            }
        min_x = min(cell.x for cell in self.known_cells) * resolution
        max_x = (max(cell.x for cell in self.known_cells) + 1) * resolution
        min_y = min(cell.y for cell in self.known_cells) * resolution
        max_y = (max(cell.y for cell in self.known_cells) + 1) * resolution
        return {
            "min_x": round(min_x, 3),
            "max_x": round(max_x, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
        }

    def _build_map_payload(self) -> dict[str, Any]:
        resolution = self.config.occupancy_resolution
        occupancy_cells = [
            {
                "x": round(cell.x * resolution, 3),
                "y": round(cell.y * resolution, 3),
                "state": state,
            }
            for cell, state in sorted(self.known_cells.items())
        ]
        semantic_area_candidates = _aggregate_semantic_updates(self.semantic_updates)
        pending_ids = {record.frontier_id for record in self.pending_candidate_records}
        remembered_frontiers = [
            record.to_dict()
            for record in self.frontier_memory.records.values()
            if record.frontier_id not in pending_ids and record.status == "stored"
        ]
        return {
            "map_id": self.config.session,
            "frame": "maniskill_world",
            "resolution": float(resolution),
            "coverage": round(self._coverage(), 3),
            "mode": "interactive_manishkill_rgbd_teleport_no_nav2",
            "summary": (
                "Interactive playground using the actual ManiSkill scene and XLeRobot head RGB-D, "
                "with teleport-only movement after operator-approved LLM decisions."
            ),
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": [],
            "frontiers": [record.to_dict() for record in self.pending_candidate_records],
            "remembered_frontiers": remembered_frontiers,
            "semantic_area_candidates": semantic_area_candidates,
            "occupancy": {
                "resolution": float(resolution),
                "bounds": self._map_bounds(),
                "cells": occupancy_cells,
            },
            "artifacts": {
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "all_frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
                "guardrail_events": self.guardrail_events,
                "navigation": {
                    "mode": "maniskill_teleport_no_nav2",
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "maniskill": {
                    "env_id": self.options.env_id,
                    "robot_uid": self.options.robot_uid,
                    "render_mode": self.options.render_mode,
                    "scan_mode": self.options.scan_mode,
                    "scan_yaw_samples": self.options.scan_yaw_samples,
                    "spawn_facing": self.options.spawn_facing,
                    "spawn_yaw": self.options.spawn_yaw,
                    "depth_beam_stride": self.options.depth_beam_stride,
                    "latest_scan_cells": len(self.latest_scan_known_cells),
                    "latest_scan_range_edge_cells": len(self.latest_scan_range_edge_cells),
                    "current_frontier_candidates": len(self.pending_candidate_records),
                    "stored_memory_frontiers": len(remembered_frontiers),
                },
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }


def _depth_image_to_millimeters(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth)
    valid = array[np.isfinite(array) & (array > 0)]
    if valid.size and float(np.nanmedian(valid)) < 50.0:
        array = array.astype(np.float32) * 1000.0
    array = np.nan_to_num(array, nan=0.0, posinf=float(np.iinfo(np.uint16).max), neginf=0.0)
    return np.asarray(np.clip(array, 0, np.iinfo(np.uint16).max), dtype=np.uint16)


def _updated_mobile_base_qpos(
    current_qpos: Any,
    pose: Pose2D,
    *,
    anchor_pose: Pose2D,
    anchor_qpos: Any,
) -> Any:
    updated = current_qpos.clone() if hasattr(current_qpos, "clone") else np.array(current_qpos, copy=True)
    anchor_qpos_array = np.asarray(anchor_qpos, dtype=np.float64)
    dx_world = float(pose.x) - float(anchor_pose.x)
    dy_world = float(pose.y) - float(anchor_pose.y)
    cos_yaw = math.cos(float(anchor_pose.yaw))
    sin_yaw = math.sin(float(anchor_pose.yaw))
    dx_local = cos_yaw * dx_world + sin_yaw * dy_world
    dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
    updated[..., 0] = float(anchor_qpos_array[..., 0] + dx_local)
    updated[..., 1] = float(anchor_qpos_array[..., 1] + dy_local)
    updated[..., 2] = float(anchor_qpos_array[..., 2] + _angle_wrap(float(pose.yaw) - float(anchor_pose.yaw)))
    return updated


def _zero_mobile_base_qvel(current_qvel: Any) -> Any:
    updated = current_qvel.clone() if hasattr(current_qvel, "clone") else np.array(current_qvel, copy=True)
    updated[..., 0] = 0.0
    updated[..., 1] = 0.0
    updated[..., 2] = 0.0
    return updated


def _cell_mean_pose(cells: list[GridCell], resolution: float, *, yaw: float = 0.0) -> Pose2D:
    if not cells:
        return Pose2D(0.0, 0.0, yaw)
    return Pose2D(
        sum((cell.x + 0.5) * resolution for cell in cells) / len(cells),
        sum((cell.y + 0.5) * resolution for cell in cells) / len(cells),
        yaw,
    )


def _spawn_facing_yaw_offset(spawn_facing: str) -> float:
    offsets = {
        "front": 0.0,
        "left": -math.pi / 2.0,
        "right": math.pi / 2.0,
        "back": math.pi,
    }
    return offsets[spawn_facing]


def _resolve_manishkill_start_pose(
    current_pose: Pose2D,
    *,
    spawn_x: float | None,
    spawn_y: float | None,
    spawn_yaw: float,
    spawn_facing: str,
) -> Pose2D | None:
    if spawn_x is not None and spawn_y is not None:
        return Pose2D(float(spawn_x), float(spawn_y), float(spawn_yaw))
    yaw_offset = _spawn_facing_yaw_offset(spawn_facing) + float(spawn_yaw)
    if abs(yaw_offset) <= 1e-9:
        return None
    return Pose2D(current_pose.x, current_pose.y, _angle_wrap(current_pose.yaw + yaw_offset))


def _angle_wrap(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def _depth_summary(depth_mm: np.ndarray, *, max_range_m: float) -> dict[str, Any]:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_range_m)]
    if valid.size == 0:
        return {
            "valid_depth_fraction": 0.0,
            "depth_min_m": None,
            "depth_p25_m": None,
            "depth_median_m": None,
            "depth_p75_m": None,
            "depth_max_m": None,
        }
    return {
        "valid_depth_fraction": round(float(valid.size) / max(depth_m.size, 1), 3),
        "depth_min_m": round(float(np.nanmin(valid)), 3),
        "depth_p25_m": round(float(np.nanpercentile(valid, 25)), 3),
        "depth_median_m": round(float(np.nanmedian(valid)), 3),
        "depth_p75_m": round(float(np.nanpercentile(valid, 75)), 3),
        "depth_max_m": round(float(np.nanmax(valid)), 3),
    }


def _rgb_array_to_data_url(rgb: Any) -> str:
    if rgb is None:
        return ""
    try:
        from PIL import Image as PILImage
    except Exception:
        return ""
    array = np.asarray(rgb)
    if array.ndim == 4:
        array = np.squeeze(array, axis=0)
    if array.ndim != 3:
        return ""
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.asarray(np.clip(array, 0, 255), dtype=np.uint8)
    image = PILImage.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _frontier_memory_prompt_context(memory: FrontierMemory) -> dict[str, Any]:
    records = sorted(memory.records.values(), key=lambda record: record.frontier_id)
    return {
        "active_frontier_id": memory.active_frontier_id,
        "frontier_status": [
            {
                "frontier_id": record.frontier_id,
                "status": record.status,
                "discovered_step": record.discovered_step,
                "last_seen_step": record.last_seen_step,
                "attempt_count": record.attempt_count,
                "visit_count": record.visit_count,
                "currently_visible": record.currently_visible,
                "llm_memory_priority": record.llm_memory_priority,
            }
            for record in records
        ],
        "return_waypoints": list(memory.return_waypoints.values()),
        "note": "Status-only memory context. Choose only from Frontier Information.",
    }


def _navigation_map_data_url(
    *,
    known_cells: dict[GridCell, str],
    resolution: float,
    trajectory: list[dict[str, Any]],
    robot_pose: Pose2D,
    candidate_records: list[FrontierRecord],
    remembered_records: list[FrontierRecord],
) -> str:
    try:
        from PIL import Image as PILImage
        from PIL import ImageDraw
    except Exception:
        return ""
    width_px = 1000
    height_px = 760
    pad_px = 36
    poses: list[Pose2D] = [robot_pose]
    for item in trajectory:
        if isinstance(item, dict):
            poses.append(Pose2D(float(item.get("x", robot_pose.x)), float(item.get("y", robot_pose.y)), float(item.get("yaw", 0.0))))
    for record in candidate_records + remembered_records:
        poses.extend([record.nav_pose, record.centroid_pose])
    if known_cells:
        min_x = min(cell.x for cell in known_cells) * resolution
        max_x = (max(cell.x for cell in known_cells) + 1) * resolution
        min_y = min(cell.y for cell in known_cells) * resolution
        max_y = (max(cell.y for cell in known_cells) + 1) * resolution
    else:
        min_x = robot_pose.x - 1.0
        max_x = robot_pose.x + 1.0
        min_y = robot_pose.y - 1.0
        max_y = robot_pose.y + 1.0
    if poses:
        min_x = min(min_x, *(pose.x for pose in poses)) - 0.35
        max_x = max(max_x, *(pose.x for pose in poses)) + 0.35
        min_y = min(min_y, *(pose.y for pose in poses)) - 0.35
        max_y = max(max_y, *(pose.y for pose in poses)) + 0.35
    world_w = max(max_x - min_x, resolution)
    world_h = max(max_y - min_y, resolution)

    def project(x: float, y: float) -> tuple[float, float]:
        px = pad_px + ((x - min_x) / world_w) * (width_px - pad_px * 2)
        py = height_px - pad_px - ((y - min_y) / world_h) * (height_px - pad_px * 2)
        return px, py

    image = PILImage.new("RGB", (width_px, height_px), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width_px - 1, height_px - 1), fill=(250, 250, 247), outline=(218, 224, 211), width=2)
    for cell, state in sorted(known_cells.items()):
        x0, y1 = project(cell.x * resolution, cell.y * resolution)
        x1, y0 = project((cell.x + 1) * resolution, (cell.y + 1) * resolution)
        fill = (115, 125, 112) if state == "occupied" else (225, 232, 219)
        draw.rectangle((x0, y0, x1, y1), fill=fill)

    trajectory_points = [
        project(float(item.get("x", robot_pose.x)), float(item.get("y", robot_pose.y)))
        for item in trajectory
        if isinstance(item, dict)
    ]
    if len(trajectory_points) >= 2:
        draw.line(trajectory_points, fill=(79, 119, 45), width=5)

    def draw_frontier(record: FrontierRecord, *, color: tuple[int, int, int], memory: bool) -> None:
        bx, by = project(record.centroid_pose.x, record.centroid_pose.y)
        nx, ny = project(record.nav_pose.x, record.nav_pose.y)
        draw.line((bx, by, nx, ny), fill=color, width=2)
        draw.ellipse((bx - 5, by - 5, bx + 5, by + 5), outline=color, width=2)
        radius = 5 if memory else 8
        draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=color)
        label = record.frontier_id
        if record.path_cost_m is not None:
            label = f"{label} {record.path_cost_m:.1f}m"
        if memory:
            label = f"{label} memory"
        draw.text((nx + 10, ny - 12), label, fill=color)

    for record in remembered_records:
        draw_frontier(record, color=(100, 112, 103), memory=True)
    for record in candidate_records:
        draw_frontier(record, color=(49, 87, 44), memory=False)

    rx, ry = project(robot_pose.x, robot_pose.y)
    hx, hy = project(
        robot_pose.x + math.cos(robot_pose.yaw) * max(resolution * 2.5, 0.45),
        robot_pose.y + math.sin(robot_pose.yaw) * max(resolution * 2.5, 0.45),
    )
    draw.ellipse((rx - 13, ry - 13, rx + 13, ry + 13), fill=(165, 40, 32))
    draw.line((rx, ry, hx, hy), fill=(109, 15, 10), width=5)
    draw.ellipse((hx - 4, hy - 4, hx + 4, hy + 4), fill=(109, 15, 10))
    draw.text((rx + 15, ry - 16), "robot", fill=(165, 40, 32))
    draw.text((18, 16), "Navigation map: free, occupied, robot, trajectory, frontier labels", fill=(24, 35, 15))

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


INTERACTIVE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XLeRobot LLM Frontier Playground</title>
  <style>
    :root {
      --bg: #eef2e6;
      --panel: rgba(255,255,255,0.82);
      --ink: #18230f;
      --muted: #5d6b52;
      --line: rgba(24,35,15,0.14);
      --green: #31572c;
      --leaf: #4f772d;
      --gold: #b0891f;
      --red: #a52820;
      --shadow: 0 24px 54px rgba(24,35,15,0.13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 6%, rgba(79,119,45,0.24), transparent 28%),
        radial-gradient(circle at 85% 16%, rgba(176,137,31,0.22), transparent 24%),
        linear-gradient(145deg, #f6f1db 0%, #e4ecd2 50%, #d9e7da 100%);
    }
    .shell { max-width: 1760px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0; font-family: Georgia, serif; font-size: clamp(32px, 5vw, 62px); line-height: .9; max-width: 11ch; }
    .subtitle { color: var(--muted); max-width: 760px; }
    .layout { display: grid; grid-template-columns: 380px minmax(520px, 1fr) 520px; gap: 16px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 16px; box-shadow: var(--shadow); backdrop-filter: blur(18px); }
    .stack { display: grid; gap: 16px; }
    .eyebrow { font-size: 12px; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); margin-bottom: 10px; }
    .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.66); padding: 11px; }
    .key { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .value { font-weight: 750; margin-top: 4px; }
    button { border: 0; border-radius: 999px; padding: 11px 14px; font-weight: 750; cursor: pointer; }
    button.primary { color: white; background: var(--green); }
    button.secondary { color: var(--green); background: #f5f0d6; border: 1px solid rgba(79,119,45,.22); }
    button.danger { color: var(--red); background: #fee8e1; }
    .buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    #map { width: 100%; height: 760px; border-radius: 20px; border: 1px solid var(--line); background: rgba(255,255,255,.9); }
    pre, textarea { width: 100%; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.72); color: #12210f; padding: 12px; overflow: auto; }
    pre { white-space: pre-wrap; max-height: 360px; margin: 0; }
    textarea { min-height: 430px; resize: vertical; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; }
    .frontier-list { display: grid; gap: 8px; max-height: 420px; overflow: auto; }
    .frontier { border: 1px solid var(--line); border-radius: 15px; padding: 10px; background: rgba(255,255,255,.66); }
    .frontier.pending { border-color: var(--gold); background: rgba(176,137,31,.12); }
    .frontier.suppressed { opacity: .55; }
    .thumbs { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .thumbs img { width: 100%; border-radius: 14px; border: 1px solid var(--line); background: white; }
    .muted { color: var(--muted); }
    .error { color: var(--red); font-weight: 750; }
    @media (max-width: 1320px) { .layout { grid-template-columns: 1fr; } #map { height: 560px; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <div class="eyebrow">No-Nav2 LLM Frontier Playground</div>
        <h1>Inspect the prompt before the robot moves.</h1>
      </div>
      <p class="subtitle">
        The robot starts in the selected backend, performs a step-gated 360 scan, pauses at each decision, then lets you call the LLM and apply the selected frontier with direct mock motion or ManiSkill teleport motion.
      </p>
    </header>
    <div class="layout">
      <div class="stack">
        <section class="panel">
          <div class="eyebrow">Controls</div>
          <div class="buttons">
            <button class="secondary" id="reset">Reset + Scan</button>
            <button class="primary" id="call">Call LLM</button>
            <button class="primary" id="apply">Move To Selected Frontier</button>
          </div>
          <p class="muted">No Nav2 is used here. Movement is direct synthetic pose update or ManiSkill teleport to the selected frontier, followed by another 360 scan.</p>
          <div id="error" class="error"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">State</div>
          <div class="stats" id="stats"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">Frontier Information</div>
          <div id="frontiers" class="frontier-list"></div>
        </section>
        <section class="panel">
          <div class="eyebrow">Recent RGB-D Views</div>
          <div id="thumbs" class="thumbs"></div>
        </section>
      </div>
      <section class="panel">
        <div class="eyebrow">Scanned 2D Map</div>
        <svg id="map" viewBox="0 0 1000 760"></svg>
      </section>
      <div class="stack">
        <section class="panel">
          <div class="eyebrow">Prompt Sent To LLM</div>
          <textarea id="prompt" readonly></textarea>
        </section>
        <section class="panel">
          <div class="eyebrow">LLM Structured Response</div>
          <pre id="response">No response yet.</pre>
        </section>
      </div>
    </div>
  </div>
  <script>
    let state = null;

    function esc(v) {
      return String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    }
    async function post(url, payload) {
      const res = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload || {})});
      if (!res.ok) throw new Error(await res.text());
      state = await res.json();
      render();
    }
    async function refresh() {
      const res = await fetch('/api/state');
      state = await res.json();
      render();
    }
    function bounds(map) {
      return map.occupancy.bounds || {min_x:0,max_x:10,min_y:0,max_y:8};
    }
    function projector(b) {
      const pad = 32;
      const w = Math.max(b.max_x - b.min_x, 1);
      const h = Math.max(b.max_y - b.min_y, 1);
      return (p) => ({
        x: pad + ((p.x - b.min_x) / w) * (1000 - pad * 2),
        y: 760 - pad - ((p.y - b.min_y) / h) * (760 - pad * 2)
      });
    }
    function renderStats() {
      const pose = state.robot_pose || {};
      const items = [
        ['Status', state.status],
        ['Coverage', state.coverage],
        ['Pose', `${Number(pose.x || 0).toFixed(2)}, ${Number(pose.y || 0).toFixed(2)}`],
        ['Frontiers', (state.candidate_frontiers || []).length],
        ['Stored Memory', (state.map?.remembered_frontiers || []).length],
        ['Pending', state.pending_target?.frontier_id || 'none'],
        ['Provider', state.map?.artifacts?.llm_policy?.provider || 'unknown']
      ];
      document.getElementById('stats').innerHTML = items.map(([k,v]) => `<div class="stat"><div class="key">${esc(k)}</div><div class="value">${esc(v)}</div></div>`).join('');
      document.getElementById('error').textContent = state.last_error || '';
    }
    function renderFrontiers() {
      const pending = state.pending_target?.frontier_id;
      document.getElementById('frontiers').innerHTML = (state.candidate_frontiers || []).map((f) => `
        <div class="frontier ${f.frontier_id === pending ? 'pending' : ''} ${f.status === 'suppressed' ? 'suppressed' : ''}">
          <strong>${esc(f.frontier_id)}</strong> · ${esc(f.status)}<br>
          gain ${esc(f.unknown_gain)} · path ${esc(f.path_cost_m ?? 'n/a')}m · priority ${esc(f.llm_memory_priority ?? 'n/a')}<br>
          <span class="muted">${esc((f.evidence || []).slice(0, 2).join(' | '))}</span>
        </div>
      `).join('') || '<div class="muted">No active frontier information.</div>';
    }
    function renderThumbs() {
      const frames = (state.map?.keyframes || []).slice(-4);
      document.getElementById('thumbs').innerHTML = frames.map((f) => `
        <div>
          <img src="${esc(f.thumbnail_data_url)}" alt="${esc(f.frame_id)}">
          <div class="muted">${esc(f.frame_id)} · ${esc(f.description)}</div>
        </div>
      `).join('');
    }
    function renderMap() {
      const svg = document.getElementById('map');
      const map = state.map;
      if (!map) { svg.innerHTML = '<text x="40" y="60">No map.</text>'; return; }
      const project = projector(bounds(map));
      const res = map.occupancy.resolution || 0.25;
      const cells = (map.occupancy.cells || []).map((c) => {
        const p = project({x:c.x, y:c.y});
        const p2 = project({x:c.x + res, y:c.y + res});
        const fill = c.state === 'occupied' ? 'rgba(24,35,15,.58)' : 'rgba(79,119,45,.16)';
        return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x-p.x)}" height="${Math.max(2, p.y-p2.y)}" fill="${fill}"/>`;
      }).join('');
      const traj = (map.trajectory || []).map((p0) => { const p = project(p0); return `${p.x},${p.y}`; }).join(' ');
      const frontiers = (map.frontiers || []).map((f) => {
        const p = project(f.nav_pose);
        const b = f.frontier_boundary_pose ? project(f.frontier_boundary_pose) : p;
        const selected = f.frontier_id === state.pending_target?.frontier_id;
        const color = f.status === 'suppressed' ? '#71717a' : selected ? '#b0891f' : '#31572c';
        const r = selected ? 11 : 7;
        return `<circle cx="${b.x}" cy="${b.y}" r="4" fill="none" stroke="${color}" stroke-width="2" opacity=".72"><title>${esc(f.frontier_id)} boundary</title></circle>
                <line x1="${b.x}" y1="${b.y}" x2="${p.x}" y2="${p.y}" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 4" opacity=".5"/>
                <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}" opacity="${selected ? 1 : .76}"><title>${esc(f.frontier_id)} approach</title></circle>
                <text x="${p.x + 10}" y="${p.y - 8}" font-size="12" fill="${color}">${esc(f.frontier_id)}</text>`;
      }).join('');
      const remembered = (map.remembered_frontiers || []).map((f) => {
        const p = project(f.nav_pose);
        const b = f.frontier_boundary_pose ? project(f.frontier_boundary_pose) : p;
        return `<circle cx="${b.x}" cy="${b.y}" r="3.5" fill="none" stroke="#647067" stroke-width="1.5" opacity=".38"><title>${esc(f.frontier_id)} stored memory boundary</title></circle>
                <line x1="${b.x}" y1="${b.y}" x2="${p.x}" y2="${p.y}" stroke="#647067" stroke-width="1" stroke-dasharray="2 5" opacity=".28"/>
                <circle cx="${p.x}" cy="${p.y}" r="5" fill="#647067" opacity=".32"><title>${esc(f.frontier_id)} stored memory approach</title></circle>
                <text x="${p.x + 8}" y="${p.y + 14}" font-size="11" fill="#647067" opacity=".7">${esc(f.frontier_id)} memory</text>`;
      }).join('');
      const robot = project(state.robot_pose || {x:0,y:0});
      const robotYaw = Number(state.robot_pose?.yaw || 0);
      const headingWorld = {
        x: (state.robot_pose?.x || 0) + Math.cos(robotYaw) * Math.max(res * 2.5, 0.45),
        y: (state.robot_pose?.y || 0) + Math.sin(robotYaw) * Math.max(res * 2.5, 0.45),
      };
      const heading = project(headingWorld);
      svg.innerHTML = `<rect width="1000" height="760" fill="rgba(255,255,255,.92)"/>${cells}
        <polyline points="${traj}" fill="none" stroke="#4f772d" stroke-width="4" stroke-linecap="round"/>
        ${remembered}
        ${frontiers}
        <circle cx="${robot.x}" cy="${robot.y}" r="13" fill="#a52820"/>
        <line x1="${robot.x}" y1="${robot.y}" x2="${heading.x}" y2="${heading.y}" stroke="#6d0f0a" stroke-width="5" stroke-linecap="round"/>
        <circle cx="${heading.x}" cy="${heading.y}" r="4.5" fill="#6d0f0a"/>
        <text x="${robot.x + 14}" y="${robot.y - 12}" fill="#a52820" font-weight="800">robot</text>`;
    }
    function renderPromptAndResponse() {
      document.getElementById('prompt').value = state.prompt || '';
      document.getElementById('response').textContent = state.pending_decision
        ? JSON.stringify({decision: state.pending_decision, applied_memory_updates: state.applied_memory_updates}, null, 2)
        : 'No response yet.';
    }
    function render() {
      renderStats();
      renderFrontiers();
      renderThumbs();
      renderMap();
      renderPromptAndResponse();
    }
    document.getElementById('reset').onclick = () => post('/api/reset');
    document.getElementById('call').onclick = () => post('/api/call_llm');
    document.getElementById('apply').onclick = () => post('/api/apply_decision');
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class InteractiveExplorationServer:
    def __init__(self, session: Any, *, host: str, port: int) -> None:
        self.session = session
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._serve_error: BaseException | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def serve_error(self) -> BaseException | None:
        return self._serve_error

    def serve_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="interactive_exploration_http", daemon=True)
        thread.start()
        return thread

    def serve_forever(self) -> None:
        session = self.session

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send_html(INTERACTIVE_HTML)
                    return
                if self.path == "/api/state":
                    self._send_json(session.snapshot())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                if self.path == "/api/reset":
                    self._send_json(session.reset())
                    return
                if self.path == "/api/call_llm":
                    self._send_json(session.call_llm())
                    return
                if self.path == "/api/apply_decision":
                    self._send_json(session.apply_decision())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_html(self, content: str) -> None:
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self._server.serve_forever()
        except BaseException as exc:
            self._serve_error = exc
            raise

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-Nav2 web playground for inspecting LLM frontier exploration prompts and decisions."
    )
    parser.add_argument("--backend", choices=("synthetic", "maniskill"), default="synthetic")
    parser.add_argument("--repo-root", default=str(Path.home() / "XLeRobot"))
    parser.add_argument("--session", default="interactive_llm_frontier")
    parser.add_argument("--area", default="apartment")
    parser.add_argument("--source", default="operator")
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--display-yaw-offset-deg", type=float, default=None)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--build-config-idx", type=int, default=None)
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--spawn-facing", choices=("front", "left", "right", "back"), default="front")
    parser.add_argument("--scan-mode", choices=("turnaround", "front_only"), default="turnaround")
    parser.add_argument("--scan-yaw-samples", type=int, default=12)
    parser.add_argument("--depth-beam-stride", type=int, default=2)
    parser.add_argument("--teleport-z", type=float, default=None)
    parser.add_argument("--teleport-settle-steps", type=int, default=1)
    parser.add_argument("--max-frontiers", type=int, default=12)
    parser.add_argument("--occupancy-resolution", type=float, default=0.25)
    parser.add_argument("--sensor-range-m", type=float, default=10.0)
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--explorer-policy", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default="mock")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimExplorationConfig(
        repo_root=args.repo_root,
        persist_path="",
        area=args.area,
        session=args.session,
        source=args.source,
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        control_mode=args.control_mode,
        render_mode=None if args.render_mode == "none" else args.render_mode,
        shader=args.shader,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        force_reload=args.force_reload,
        occupancy_resolution=args.occupancy_resolution,
        sensor_range_m=args.sensor_range_m,
        finish_coverage_threshold=args.finish_coverage_threshold,
        max_decisions=args.max_decisions,
        explorer_policy=args.explorer_policy,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_reasoning_effort=args.llm_reasoning_effort,
        nav2_mode="simulated",
        realtime_sleep_s=0.0,
    )
    if args.backend == "maniskill":
        session = ManiSkillTeleportExplorationSession(
            config,
            ManiSkillInteractiveOptions(
                repo_root=args.repo_root,
                env_id=args.env_id,
                robot_uid=args.robot_uid,
                display_yaw_offset_deg=args.display_yaw_offset_deg,
                control_mode=args.control_mode,
                render_mode=None if args.render_mode == "none" else args.render_mode,
                shader=args.shader,
                sim_backend=args.sim_backend,
                num_envs=args.num_envs,
                force_reload=args.force_reload,
                build_config_idx=args.build_config_idx,
                spawn_x=args.spawn_x,
                spawn_y=args.spawn_y,
                spawn_yaw=args.spawn_yaw,
                spawn_facing=args.spawn_facing,
                scan_mode=args.scan_mode,
                scan_yaw_samples=args.scan_yaw_samples,
                depth_beam_stride=args.depth_beam_stride,
                teleport_z=args.teleport_z,
                teleport_settle_steps=args.teleport_settle_steps,
                max_frontiers=args.max_frontiers,
            ),
        )
    else:
        session = InteractiveNoNav2ExplorationSession(config)
    server = InteractiveExplorationServer(session, host=args.host, port=args.port)
    print(f"Interactive LLM frontier playground running at {server.url} (backend={args.backend})")
    print("Flow: inspect prompt -> Call LLM -> inspect response/target -> Move To Selected Frontier.")
    if args.open_browser:
        webbrowser.open(server.url)
    try:
        if args.backend == "maniskill":
            server_thread = server.serve_in_background()
            print("Pumping the ManiSkill/SAPIEN viewer on the main thread to keep the window responsive.")
            while server_thread.is_alive():
                if server.serve_error is not None:
                    raise server.serve_error
                pump_viewer = getattr(session, "pump_viewer", None)
                if callable(pump_viewer):
                    pump_viewer()
                time.sleep(1.0 / 30.0)
            if server.serve_error is not None:
                raise server.serve_error
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down interactive playground.")
    finally:
        server.shutdown()
        close = getattr(session, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
