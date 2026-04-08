from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig, Pose2D
from xlerobot_agent.exploration_ui import ExplorationReviewServer, LocalExplorationUIController
from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, ModelConfig
from xlerobot_agent.prompts import (
    build_exploration_policy_system_prompt,
    build_exploration_policy_user_prompt,
)


@dataclass
class SimExplorationConfig:
    repo_root: str
    persist_path: str
    env_id: str = "SceneManipulation-v1"
    robot_uid: str = "xlerobot"
    control_mode: str = "pd_joint_delta_pos_dual_arm"
    render_mode: str | None = "human"
    shader: str = "default"
    sim_backend: str = "auto"
    num_envs: int = 1
    force_reload: bool = False
    area: str = "apartment"
    session: str = "house_v1"
    source: str = "operator"
    occupancy_resolution: float = 0.25
    max_control_steps: int | None = None
    max_episode_steps: int | None = None
    show_cameras: bool = True
    use_rerun: bool = False
    camera_log_stride: int = 2
    realtime_sleep_s: float = 0.01
    explorer_policy: str = "llm"
    llm_provider: str = "mock"
    llm_model: str = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1200
    llm_reasoning_effort: str | None = None
    trace_policy_stdout: bool = False
    trace_llm_stdout: bool = False
    review_host: str = "127.0.0.1"
    review_port: int = 8770
    serve_review_ui: bool = False
    sensor_range_m: float = 10.0
    robot_radius_m: float = 0.22
    finish_coverage_threshold: float = 0.96
    max_decisions: int = 32
    nav2_planner_id: str = "GridBased"
    nav2_controller_id: str = "FollowPath"
    nav2_behavior_tree: str = "navigate_to_pose_w_replanning_and_recovery.xml"
    nav2_recovery_enabled: bool = True


@dataclass(frozen=True, order=True)
class GridCell:
    x: int
    y: int

    def center_pose(self, resolution: float, *, yaw: float = 0.0) -> Pose2D:
        return Pose2D((self.x + 0.5) * resolution, (self.y + 0.5) * resolution, yaw)

    def to_xy(self, resolution: float) -> tuple[float, float]:
        pose = self.center_pose(resolution)
        return pose.x, pose.y


@dataclass(frozen=True)
class SubAreaSpec:
    area_id: str
    label: str
    polygon_2d: tuple[tuple[float, float], ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RoomSpec:
    region_id: str
    label: str
    polygon_2d: tuple[tuple[float, float], ...]
    adjacency: tuple[str, ...]
    objects: tuple[str, ...]
    descriptions: tuple[str, ...]
    entry_pose: Pose2D
    center_pose: Pose2D
    sub_areas: tuple[SubAreaSpec, ...] = tuple()


@dataclass(frozen=True)
class WorldObject:
    object_id: str
    label: str
    room_id: str
    cell: GridCell
    description: str


@dataclass
class ApartmentScenario:
    scenario_id: str
    layout_id: str
    resolution: float
    width_cells: int
    height_cells: int
    start_cell: GridCell
    free_cells: set[GridCell]
    obstacle_cells: set[GridCell]
    room_by_cell: dict[GridCell, str]
    rooms: dict[str, RoomSpec]
    objects: list[WorldObject]

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell.x < self.width_cells and 0 <= cell.y < self.height_cells

    def is_free(self, cell: GridCell) -> bool:
        return cell in self.free_cells

    def is_occupied(self, cell: GridCell) -> bool:
        return not self.in_bounds(cell) or cell in self.obstacle_cells or cell not in self.free_cells

    def room_for_cell(self, cell: GridCell) -> str | None:
        return self.room_by_cell.get(cell)

    def room_for_pose(self, pose: Pose2D) -> str | None:
        return self.room_for_cell(self.world_to_cell(pose.x, pose.y))

    def world_to_cell(self, x: float, y: float) -> GridCell:
        return GridCell(int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution)))

    def bounds(self) -> dict[str, float]:
        return {
            "min_x": 0.0,
            "max_x": round(self.width_cells * self.resolution, 3),
            "min_y": 0.0,
            "max_y": round(self.height_cells * self.resolution, 3),
        }

    def total_free_cells(self) -> int:
        return len(self.free_cells)


@dataclass
class ScanResult:
    observed_free: set[GridCell]
    observed_occupied: set[GridCell]
    range_edge_frontiers: set[GridCell]
    visible_objects: list[str]
    visible_room_ids: list[str]
    point_count: int
    depth_min_m: float
    depth_max_m: float
    description: str
    thumbnail_data_url: str


@dataclass
class FrontierCandidate:
    frontier_id: str | None
    member_cells: tuple[GridCell, ...]
    nav_cell: GridCell
    centroid_cell: GridCell
    nav_pose: Pose2D
    centroid_pose: Pose2D
    unknown_gain: int
    sensor_range_edge: bool
    room_hint: str | None
    evidence: list[str]
    currently_visible: bool = True
    path_cost_m: float | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "nav_pose": self.nav_pose.to_dict(),
            "centroid_pose": self.centroid_pose.to_dict(),
            "unknown_gain": self.unknown_gain,
            "sensor_range_edge": self.sensor_range_edge,
            "room_hint": self.room_hint,
            "currently_visible": self.currently_visible,
            "path_cost_m": None if self.path_cost_m is None else round(self.path_cost_m, 3),
            "evidence": list(self.evidence),
        }


@dataclass
class FrontierRecord:
    frontier_id: str
    nav_pose: Pose2D
    centroid_pose: Pose2D
    status: str
    discovered_step: int
    last_seen_step: int
    unknown_gain: int
    sensor_range_edge: bool
    room_hint: str | None
    evidence: list[str] = field(default_factory=list)
    attempt_count: int = 0
    visit_count: int = 0
    path_cost_m: float | None = None
    currently_visible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "nav_pose": self.nav_pose.to_dict(),
            "centroid_pose": self.centroid_pose.to_dict(),
            "status": self.status,
            "discovered_step": self.discovered_step,
            "last_seen_step": self.last_seen_step,
            "unknown_gain": self.unknown_gain,
            "sensor_range_edge": self.sensor_range_edge,
            "room_hint": self.room_hint,
            "evidence": list(self.evidence),
            "attempt_count": self.attempt_count,
            "visit_count": self.visit_count,
            "path_cost_m": None if self.path_cost_m is None else round(self.path_cost_m, 3),
            "currently_visible": self.currently_visible,
        }


@dataclass
class ExplorationDecision:
    decision_type: str
    selected_frontier_id: str | None
    selected_return_waypoint_id: str | None
    frontier_ids_to_store: list[str]
    exploration_complete: bool
    reasoning_summary: str
    semantic_updates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "selected_frontier_id": self.selected_frontier_id,
            "selected_return_waypoint_id": self.selected_return_waypoint_id,
            "frontier_ids_to_store": list(self.frontier_ids_to_store),
            "exploration_complete": self.exploration_complete,
            "reasoning_summary": self.reasoning_summary,
            "semantic_updates": json.loads(json.dumps(self.semantic_updates)),
        }


@dataclass(frozen=True)
class Nav2GoalRequest:
    goal_id: str
    goal_type: str
    target_pose: Pose2D
    planner_id: str
    controller_id: str
    behavior_tree: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "target_pose": self.target_pose.to_dict(),
            "planner_id": self.planner_id,
            "controller_id": self.controller_id,
            "behavior_tree": self.behavior_tree,
            "metadata": json.loads(json.dumps(self.metadata)),
        }


@dataclass(frozen=True)
class Nav2GoalValidation:
    accepted: bool
    goal: Nav2GoalRequest
    normalized_pose: Pose2D | None
    goal_cell: GridCell | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "goal": self.goal.to_dict(),
            "normalized_pose": None if self.normalized_pose is None else self.normalized_pose.to_dict(),
            "goal_cell": None if self.goal_cell is None else {"x": self.goal_cell.x, "y": self.goal_cell.y},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Nav2PlanResult:
    status: str
    goal: Nav2GoalRequest
    planner_id: str
    reason: str
    goal_cell: GridCell | None
    path_cells: tuple[GridCell, ...] = tuple()
    path_length_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal.to_dict(),
            "planner_id": self.planner_id,
            "reason": self.reason,
            "goal_cell": None if self.goal_cell is None else {"x": self.goal_cell.x, "y": self.goal_cell.y},
            "path_length_m": round(self.path_length_m, 3),
            "waypoint_count": len(self.path_cells),
            "path": [
                {"x": point.x, "y": point.y}
                for point in self.path_cells
            ],
        }


@dataclass(frozen=True)
class Nav2NavigateResult:
    status: str
    goal: Nav2GoalRequest
    plan: Nav2PlanResult
    reached_pose: Pose2D | None
    travelled_distance_m: float
    reason: str
    feedback_samples: tuple[dict[str, Any], ...] = tuple()
    recovery_events: tuple[dict[str, Any], ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal.to_dict(),
            "plan": self.plan.to_dict(),
            "reached_pose": None if self.reached_pose is None else self.reached_pose.to_dict(),
            "travelled_distance_m": round(self.travelled_distance_m, 3),
            "reason": self.reason,
            "feedback_samples": list(self.feedback_samples),
            "recovery_events": list(self.recovery_events),
        }


class Nav2NavigationModule:
    def validate_goal(self, goal: Nav2GoalRequest) -> Nav2GoalValidation:
        raise NotImplementedError

    def compute_path(self, goal: Nav2GoalRequest, *, record: bool = True) -> Nav2PlanResult:
        raise NotImplementedError

    def navigate_to_pose(self, goal: Nav2GoalRequest) -> Nav2NavigateResult:
        raise NotImplementedError

    def recover(self, goal: Nav2GoalRequest, *, reason: str) -> dict[str, Any]:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError


class SimulatedNav2NavigationModule(Nav2NavigationModule):
    def __init__(
        self,
        config: SimExplorationConfig,
        scenario: ApartmentScenario,
        *,
        get_current_cell: Callable[[], GridCell],
        get_current_yaw: Callable[[], float],
        known_free_cells: Callable[[], set[GridCell]],
        on_motion_step: Callable[[GridCell, GridCell, Nav2GoalRequest, int, int], None],
        on_runtime_obstacle: Callable[[GridCell], None],
        budget_exhausted: Callable[[], bool],
    ) -> None:
        self.config = config
        self.scenario = scenario
        self._get_current_cell = get_current_cell
        self._get_current_yaw = get_current_yaw
        self._known_free_cells = known_free_cells
        self._on_motion_step = on_motion_step
        self._on_runtime_obstacle = on_runtime_obstacle
        self._budget_exhausted = budget_exhausted
        self._goal_history: list[dict[str, Any]] = []
        self._plan_history: list[dict[str, Any]] = []
        self._recovery_history: list[dict[str, Any]] = []

    def validate_goal(self, goal: Nav2GoalRequest) -> Nav2GoalValidation:
        goal_cell = self.scenario.world_to_cell(goal.target_pose.x, goal.target_pose.y)
        if not self.scenario.in_bounds(goal_cell):
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=None,
                goal_cell=goal_cell,
                reason="goal lies outside the known map bounds",
            )
        normalized_pose = goal_cell.center_pose(self.scenario.resolution, yaw=goal.target_pose.yaw)
        known_free = self._known_free_cells()
        if goal_cell not in known_free:
            return Nav2GoalValidation(
                accepted=False,
                goal=goal,
                normalized_pose=normalized_pose,
                goal_cell=goal_cell,
                reason="goal is not currently reachable known free space for Nav2",
            )
        return Nav2GoalValidation(
            accepted=True,
            goal=goal,
            normalized_pose=normalized_pose,
            goal_cell=goal_cell,
            reason="goal accepted by simulated Nav2 goal checker",
        )

    def compute_path(self, goal: Nav2GoalRequest, *, record: bool = True) -> Nav2PlanResult:
        validation = self.validate_goal(goal)
        if not validation.accepted:
            result = Nav2PlanResult(
                status="rejected",
                goal=goal,
                planner_id=goal.planner_id,
                reason=validation.reason,
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        path_cells = _search_known_safe_path(
            self._get_current_cell(),
            validation.goal_cell,
            self._known_free_cells(),
        )
        if not path_cells:
            result = Nav2PlanResult(
                status="failed",
                goal=goal,
                planner_id=goal.planner_id,
                reason="planner could not find a known-safe path to the goal",
                goal_cell=validation.goal_cell,
            )
            if record:
                self._plan_history.append(result.to_dict())
            return result
        result = Nav2PlanResult(
            status="succeeded",
            goal=goal,
            planner_id=goal.planner_id,
            reason="planner computed a path on the current known-free occupancy map",
            goal_cell=validation.goal_cell,
            path_cells=tuple(path_cells),
            path_length_m=max(len(path_cells) - 1, 0) * self.scenario.resolution,
        )
        if record:
            self._plan_history.append(result.to_dict())
        return result

    def navigate_to_pose(self, goal: Nav2GoalRequest) -> Nav2NavigateResult:
        self._goal_history.append(goal.to_dict())
        plan = self.compute_path(goal, record=True)
        recovery_events: list[dict[str, Any]] = []
        if plan.status != "succeeded":
            if self.config.nav2_recovery_enabled:
                recovery_events.append(self.recover(goal, reason=plan.reason))
            return Nav2NavigateResult(
                status="failed",
                goal=goal,
                plan=plan,
                reached_pose=None,
                travelled_distance_m=0.0,
                reason=plan.reason,
                recovery_events=tuple(recovery_events),
            )

        feedback_samples: list[dict[str, Any]] = []
        travelled_distance_m = 0.0
        path_cells = list(plan.path_cells)
        for index, (previous, nxt) in enumerate(zip(path_cells, path_cells[1:]), start=1):
            if self._budget_exhausted():
                reason = "Nav2 execution stopped because the control-step budget was exhausted"
                if self.config.nav2_recovery_enabled:
                    recovery_events.append(self.recover(goal, reason=reason))
                return Nav2NavigateResult(
                    status="failed",
                    goal=goal,
                    plan=plan,
                    reached_pose=previous.center_pose(self.scenario.resolution, yaw=self._get_current_yaw()),
                    travelled_distance_m=travelled_distance_m,
                    reason=reason,
                    feedback_samples=tuple(feedback_samples),
                    recovery_events=tuple(recovery_events),
                )
            if self.scenario.is_occupied(nxt):
                self._on_runtime_obstacle(nxt)
                reason = "Nav2 execution encountered a runtime obstacle on the current path"
                if self.config.nav2_recovery_enabled:
                    recovery_events.append(self.recover(goal, reason=reason))
                return Nav2NavigateResult(
                    status="failed",
                    goal=goal,
                    plan=plan,
                    reached_pose=previous.center_pose(self.scenario.resolution, yaw=self._get_current_yaw()),
                    travelled_distance_m=travelled_distance_m,
                    reason=reason,
                    feedback_samples=tuple(feedback_samples),
                    recovery_events=tuple(recovery_events),
                )
            self._on_motion_step(previous, nxt, goal, index, len(path_cells))
            travelled_distance_m += self.scenario.resolution
            remaining_distance_m = max((len(path_cells) - index - 1), 0) * self.scenario.resolution
            feedback_samples.append(
                {
                    "step_index": index,
                    "remaining_distance_m": round(remaining_distance_m, 3),
                    "current_pose": nxt.center_pose(self.scenario.resolution).to_dict(),
                    "status": "moving",
                }
            )

        reached_pose = path_cells[-1].center_pose(self.scenario.resolution, yaw=goal.target_pose.yaw)
        return Nav2NavigateResult(
            status="succeeded",
            goal=goal,
            plan=plan,
            reached_pose=reached_pose,
            travelled_distance_m=travelled_distance_m,
            reason="Nav2 reached the requested goal pose",
            feedback_samples=tuple(feedback_samples),
            recovery_events=tuple(recovery_events),
        )

    def recover(self, goal: Nav2GoalRequest, *, reason: str) -> dict[str, Any]:
        event = {
            "goal_id": goal.goal_id,
            "behavior_tree": goal.behavior_tree,
            "recovery_action": "spin_and_rescan",
            "reason": reason,
            "timestamp": time.time(),
        }
        self._recovery_history.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "module": "simulated_nav2",
            "planner_id": self.config.nav2_planner_id,
            "controller_id": self.config.nav2_controller_id,
            "behavior_tree": self.config.nav2_behavior_tree,
            "goals": list(self._goal_history),
            "plans": list(self._plan_history),
            "recoveries": list(self._recovery_history),
        }

class FrontierMemory:
    def __init__(self, resolution: float) -> None:
        self.resolution = resolution
        self.records: dict[str, FrontierRecord] = {}
        self.active_frontier_id: str | None = None
        self.return_waypoints: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def upsert_candidates(self, candidates: list[FrontierCandidate], *, step_index: int) -> list[FrontierRecord]:
        visible_ids: set[str] = set()
        for candidate in candidates:
            match_id, distance_m = self._find_match_id(candidate)
            if match_id is not None:
                record = self.records[match_id]
                if record.status in {"completed", "failed"} and distance_m <= self._dedupe_tolerance_m():
                    if (
                        distance_m <= self.resolution * 2.0
                        and candidate.unknown_gain <= record.unknown_gain + 1
                        and not candidate.sensor_range_edge
                    ):
                        continue
                    match_id = None
            if match_id is None:
                frontier_id = f"frontier_{self._counter:03d}"
                self._counter += 1
                record = FrontierRecord(
                    frontier_id=frontier_id,
                    nav_pose=candidate.nav_pose,
                    centroid_pose=candidate.centroid_pose,
                    status="stored",
                    discovered_step=step_index,
                    last_seen_step=step_index,
                    unknown_gain=candidate.unknown_gain,
                    sensor_range_edge=candidate.sensor_range_edge,
                    room_hint=candidate.room_hint,
                    evidence=list(candidate.evidence),
                    currently_visible=True,
                )
                self.records[frontier_id] = record
            else:
                record = self.records[match_id]
                record.nav_pose = candidate.nav_pose
                record.centroid_pose = candidate.centroid_pose
                record.last_seen_step = step_index
                record.unknown_gain = max(record.unknown_gain, candidate.unknown_gain)
                record.sensor_range_edge = record.sensor_range_edge or candidate.sensor_range_edge
                record.room_hint = candidate.room_hint or record.room_hint
                record.currently_visible = True
                record.evidence = _dedupe_text(record.evidence + candidate.evidence)
                if record.status not in {"active", "failed", "completed"}:
                    record.status = "stored"
            visible_ids.add(record.frontier_id)

        for record in self.records.values():
            if record.frontier_id not in visible_ids:
                record.currently_visible = False
        return [self.records[item] for item in visible_ids]

    def candidate_records(self) -> list[FrontierRecord]:
        return [
            record
            for record in self.records.values()
            if record.status not in {"completed", "failed"}
        ]

    def activate(self, frontier_id: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        if self.active_frontier_id and self.active_frontier_id != frontier_id:
            previous = self.records.get(self.active_frontier_id)
            if previous is not None and previous.status == "active":
                previous.status = "stored"
        self.active_frontier_id = frontier_id
        record.status = "active"
        record.attempt_count += 1
        return record

    def complete(self, frontier_id: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        record.status = "completed"
        record.visit_count += 1
        if self.active_frontier_id == frontier_id:
            self.active_frontier_id = None
        return record

    def fail(self, frontier_id: str, reason: str) -> FrontierRecord | None:
        record = self.records.get(frontier_id)
        if record is None:
            return None
        record.status = "failed"
        record.evidence = _dedupe_text(record.evidence + [reason])
        if self.active_frontier_id == frontier_id:
            self.active_frontier_id = None
        return record

    def remember_return_waypoint(self, *, room_id: str | None, pose: Pose2D, step_index: int, reason: str) -> dict[str, Any]:
        prefix = (room_id or "unknown").replace("region_", "")
        waypoint_id = f"return_{prefix}_{step_index:03d}"
        entry = {
            "waypoint_id": waypoint_id,
            "room_id": room_id,
            "pose": pose.to_dict(),
            "step_index": step_index,
            "reason": reason,
        }
        self.return_waypoints[waypoint_id] = entry
        return entry

    def get_return_waypoint(self, waypoint_id: str | None) -> dict[str, Any] | None:
        if waypoint_id is None:
            return None
        return self.return_waypoints.get(waypoint_id)

    def snapshot(self) -> dict[str, Any]:
        stored = [record.to_dict() for record in self.records.values() if record.status == "stored"]
        active = None if self.active_frontier_id is None else self.records[self.active_frontier_id].to_dict()
        visited = [record.to_dict() for record in self.records.values() if record.visit_count > 0]
        failed = [record.to_dict() for record in self.records.values() if record.status == "failed"]
        completed = [record.to_dict() for record in self.records.values() if record.status == "completed"]
        return {
            "active_frontier": active,
            "stored_frontiers": stored,
            "visited_frontiers": visited,
            "failed_frontiers": failed,
            "completed_frontiers": completed,
            "return_waypoints": list(self.return_waypoints.values()),
        }

    def _dedupe_tolerance_m(self) -> float:
        return max(self.resolution * 4.0, 0.75)

    def _find_match_id(self, candidate: FrontierCandidate) -> tuple[str | None, float]:
        best_id = None
        best_distance = 1e9
        for record in self.records.values():
            distance = _pose_distance_m(record.nav_pose, candidate.nav_pose)
            if distance < best_distance:
                best_distance = distance
                best_id = record.frontier_id
        if best_id is None or best_distance > self._dedupe_tolerance_m():
            return None, best_distance
        return best_id, best_distance


class ExplorationLLMPolicy:
    def __init__(self, config: SimExplorationConfig) -> None:
        self.config = config
        self.system_prompt = build_exploration_policy_system_prompt()
        self.router: AgentLLMRouter | None = None
        if self.config.explorer_policy == "llm" and self.config.llm_provider != "mock":
            model = ModelConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                reasoning_effort=self.config.llm_reasoning_effort,
            )
            self.router = AgentLLMRouter(
                AgentModelSuite(
                    planner=model,
                    critic=model,
                    coder=model,
                )
            )

    def decide(
        self,
        *,
        prompt_payload: dict[str, Any],
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
        coverage: float,
        current_room_id: str | None,
    ) -> tuple[ExplorationDecision, dict[str, Any]]:
        heuristic = self._heuristic_decision(frontiers, return_waypoints, coverage, current_room_id)
        user_prompt = build_exploration_policy_user_prompt(prompt_payload)
        trace: dict[str, Any] = {
            "mode": self.config.explorer_policy,
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "prompt": user_prompt,
        }

        if self.config.explorer_policy != "llm":
            trace["response"] = heuristic.to_dict()
            return heuristic, trace

        if self.router is None:
            trace["response"] = heuristic.to_dict()
            return heuristic, trace

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *[
                        {"type": "image_url", "image_url": {"url": item["thumbnail_data_url"]}}
                        for item in prompt_payload.get("recent_views", [])[-2:]
                        if item.get("thumbnail_data_url")
                    ],
                ],
            },
        ]
        parsed, llm_trace = self.router.complete_json_messages(
            config=self.router.model_suite.planner,
            messages=messages,
        )
        trace["llm_trace"] = {
            "duration_s": llm_trace.duration_s,
            "error": llm_trace.error,
            "response_text": llm_trace.response_text,
        }
        if parsed is None:
            trace["response"] = heuristic.to_dict()
            trace["fallback_reason"] = "llm_parse_failure"
            return heuristic, trace

        decision = self._parse_model_decision(parsed, frontiers, return_waypoints)
        if decision is None:
            trace["response"] = heuristic.to_dict()
            trace["fallback_reason"] = "llm_invalid_decision"
            return heuristic, trace
        trace["response"] = decision.to_dict()
        return decision, trace

    def _heuristic_decision(
        self,
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
        coverage: float,
        current_room_id: str | None,
    ) -> ExplorationDecision:
        reachable = [record for record in frontiers if record.path_cost_m is not None]
        if not reachable:
            return ExplorationDecision(
                decision_type="finish",
                selected_frontier_id=None,
                selected_return_waypoint_id=None,
                frontier_ids_to_store=[],
                exploration_complete=True,
                reasoning_summary=(
                    "No reachable frontier remains in memory, so the exploration loop should finish."
                ),
                semantic_updates=[],
            )

        def score(record: FrontierRecord) -> float:
            distance_penalty = record.path_cost_m or 0.0
            novelty_bonus = 1.2 if record.currently_visible else 0.4
            range_bonus = 1.8 if record.sensor_range_edge else 0.0
            revisit_penalty = min(record.attempt_count * 0.8, 2.4)
            room_bonus = 0.9 if record.room_hint and record.room_hint != current_room_id else 0.0
            return record.unknown_gain * 1.4 + novelty_bonus + range_bonus + room_bonus - distance_penalty * 0.16 - revisit_penalty

        ranked = sorted(reachable, key=score, reverse=True)
        best = ranked[0]
        hallway_waypoint = next(
            (
                item
                for item in reversed(return_waypoints)
                if item.get("room_id") == "region_hallway"
            ),
            None,
        )
        selected_return_waypoint_id = None
        if hallway_waypoint and current_room_id not in {None, "region_hallway"} and best.room_hint != current_room_id:
            selected_return_waypoint_id = str(hallway_waypoint["waypoint_id"])
        semantic_updates: list[dict[str, Any]] = []
        if best.room_hint:
            semantic_updates.append(
                {
                    "label": best.room_hint.replace("region_", ""),
                    "kind": "room_hint",
                    "target_id": best.frontier_id,
                    "confidence": 0.58 if best.sensor_range_edge else 0.51,
                    "evidence": best.evidence[:3],
                }
            )
        return ExplorationDecision(
            decision_type="revisit_frontier" if not best.currently_visible else "explore_frontier",
            selected_frontier_id=best.frontier_id,
            selected_return_waypoint_id=selected_return_waypoint_id,
            frontier_ids_to_store=[record.frontier_id for record in ranked[1:] if record.frontier_id],
            exploration_complete=coverage >= self.config.finish_coverage_threshold and len(reachable) <= 1,
            reasoning_summary=(
                f"Select {best.frontier_id} because it offers the best balance of unknown gain, "
                f"room novelty, and travel cost."
            ),
            semantic_updates=semantic_updates,
        )

    def _parse_model_decision(
        self,
        payload: dict[str, Any],
        frontiers: list[FrontierRecord],
        return_waypoints: list[dict[str, Any]],
    ) -> ExplorationDecision | None:
        valid_frontier_ids = {record.frontier_id for record in frontiers}
        valid_waypoint_ids = {str(item["waypoint_id"]) for item in return_waypoints}
        decision_type = str(payload.get("decision_type", "")).strip()
        if decision_type not in {"explore_frontier", "revisit_frontier", "finish"}:
            return None
        selected_frontier_id = payload.get("selected_frontier_id")
        if decision_type != "finish" and selected_frontier_id not in valid_frontier_ids:
            return None
        if decision_type == "finish":
            selected_frontier_id = None
        selected_return_waypoint_id = payload.get("selected_return_waypoint_id")
        if selected_return_waypoint_id is not None and selected_return_waypoint_id not in valid_waypoint_ids:
            selected_return_waypoint_id = None
        frontier_ids_to_store = [
            str(item)
            for item in payload.get("frontier_ids_to_store", [])
            if str(item) in valid_frontier_ids and str(item) != selected_frontier_id
        ]
        semantic_updates = [
            item
            for item in payload.get("semantic_updates", [])
            if isinstance(item, dict)
        ][:6]
        return ExplorationDecision(
            decision_type=decision_type,
            selected_frontier_id=selected_frontier_id,
            selected_return_waypoint_id=selected_return_waypoint_id,
            frontier_ids_to_store=frontier_ids_to_store,
            exploration_complete=bool(payload.get("exploration_complete", decision_type == "finish")),
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip() or "Model selected the next exploration action.",
            semantic_updates=json.loads(json.dumps(semantic_updates)),
        )


class _ApartmentExplorationSession:
    def __init__(self, config: SimExplorationConfig, backend: ExplorationBackend, task_id: str) -> None:
        self.config = config
        self.backend = backend
        self.task_id = task_id
        self.scenario = _build_simple_apartment(config.occupancy_resolution)
        self.policy = ExplorationLLMPolicy(config)
        self.frontier_memory = FrontierMemory(config.occupancy_resolution)
        self.current_cell = self.scenario.start_cell
        self.current_yaw = 0.0
        self.known_cells: dict[GridCell, str] = {}
        self.range_edge_cells: set[GridCell] = set()
        self.trajectory: list[dict[str, Any]] = [self.current_cell.center_pose(self.scenario.resolution).to_dict()]
        self.keyframes: list[dict[str, Any]] = []
        self.room_frames: dict[str, list[str]] = {}
        self.room_objects_seen: dict[str, set[str]] = {room_id: set() for room_id in self.scenario.rooms}
        self.room_descriptions: dict[str, list[str]] = {room_id: [] for room_id in self.scenario.rooms}
        self.decision_log: list[dict[str, Any]] = []
        self.guardrail_events: list[dict[str, Any]] = []
        self.semantic_updates: list[dict[str, Any]] = []
        self.total_distance_m = 0.0
        self.control_steps = 0
        self.decision_index = 0
        self.nav2_goal_counter = 0
        self.nav2 = SimulatedNav2NavigationModule(
            config,
            self.scenario,
            get_current_cell=lambda: self.current_cell,
            get_current_yaw=lambda: self.current_yaw,
            known_free_cells=self._known_free_cells,
            on_motion_step=self._on_nav2_motion_step,
            on_runtime_obstacle=self._on_nav2_runtime_obstacle,
            budget_exhausted=self._budget_exhausted,
        )

    def run(self) -> dict[str, Any]:
        start_pose = self.current_cell.center_pose(self.scenario.resolution)
        self.frontier_memory.remember_return_waypoint(
            room_id=self.scenario.room_for_cell(self.current_cell),
            pose=start_pose,
            step_index=0,
            reason="initial_pose",
        )
        self._perform_scan(
            full_turnaround=True,
            capture_frame=True,
            reason="initial_turnaround_scan",
        )

        while self.decision_index < self.config.max_decisions:
            if self._budget_exhausted():
                self.guardrail_events.append(
                    {
                        "type": "budget_exhausted",
                        "control_steps": self.control_steps,
                        "decision_index": self.decision_index,
                    }
                )
                break

            self.decision_index += 1
            visible_candidates = self._detect_frontier_candidates()
            self.frontier_memory.upsert_candidates(visible_candidates, step_index=self.decision_index)
            candidate_records = self._refresh_candidate_paths()
            prompt_payload = self._build_prompt_payload(candidate_records)
            coverage = self._coverage()
            decision, trace = self.policy.decide(
                prompt_payload=prompt_payload,
                frontiers=candidate_records,
                return_waypoints=list(self.frontier_memory.return_waypoints.values()),
                coverage=coverage,
                current_room_id=self.scenario.room_for_cell(self.current_cell),
            )
            decision = self._apply_finish_guardrail(decision, candidate_records)
            self.semantic_updates.extend(decision.semantic_updates)
            self._log_policy_step(prompt_payload, decision, trace)
            if decision.decision_type == "finish" or decision.exploration_complete and not candidate_records:
                break

            if decision.selected_return_waypoint_id:
                waypoint = self.frontier_memory.get_return_waypoint(decision.selected_return_waypoint_id)
                if waypoint is not None:
                    target_pose = waypoint["pose"]
                    return_goal = self._make_nav2_goal(
                        Pose2D(float(target_pose["x"]), float(target_pose["y"]), float(target_pose.get("yaw", 0.0))),
                        goal_type="return_waypoint",
                        reason=f"return_waypoint::{waypoint['waypoint_id']}",
                    )
                    return_result = self.nav2.navigate_to_pose(return_goal)
                    if return_result.status != "succeeded":
                        self.guardrail_events.append(
                            {
                                "type": "return_waypoint_failed",
                                "waypoint_id": waypoint["waypoint_id"],
                                "nav2_result": return_result.to_dict(),
                            }
                        )

            if not decision.selected_frontier_id:
                break

            record = self.frontier_memory.activate(decision.selected_frontier_id)
            if record is None:
                self.guardrail_events.append(
                    {
                        "type": "missing_frontier",
                        "frontier_id": decision.selected_frontier_id,
                    }
                )
                continue

            frontier_goal = self._make_nav2_goal(
                record.nav_pose,
                goal_type="frontier",
                reason=f"frontier::{record.frontier_id}",
            )
            nav_result = self.nav2.navigate_to_pose(frontier_goal)
            if nav_result.status != "succeeded":
                self.frontier_memory.fail(record.frontier_id, nav_result.reason)
                self._push_progress_update(
                    message=f"Failed to reach {record.frontier_id}: {nav_result.reason}",
                    frontier_id=record.frontier_id,
                )
                continue

            self._perform_scan(
                full_turnaround=True,
                capture_frame=True,
                reason=f"arrive_frontier::{record.frontier_id}",
            )
            self.frontier_memory.complete(record.frontier_id)
            self.frontier_memory.remember_return_waypoint(
                room_id=self.scenario.room_for_cell(self.current_cell),
                pose=self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw),
                step_index=self.decision_index,
                reason=f"completed_frontier::{record.frontier_id}",
            )
            self._push_progress_update(
                message=f"Explored {record.frontier_id} from room `{self.scenario.room_for_cell(self.current_cell)}`.",
                frontier_id=record.frontier_id,
            )

        return self._build_map_payload()

    def _budget_exhausted(self) -> bool:
        if self.config.max_control_steps is not None and self.control_steps >= self.config.max_control_steps:
            return True
        if self.config.max_episode_steps is not None and self.decision_index >= self.config.max_episode_steps:
            return True
        return False

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
        self._sleep()

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
            if len(unknown_neighbors) >= 2 or (cell in self.range_edge_cells and unknown_neighbors):
                frontier_cells.add(cell)
                unknown_neighbors_by_frontier[cell] = unknown_neighbors

        clusters: list[list[GridCell]] = []
        visited: set[GridCell] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: list[GridCell] = []
            queue = deque([cell])
            visited.add(cell)
            while queue:
                current = queue.popleft()
                cluster.append(current)
                for neighbor in _neighbors8(current):
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        candidates: list[FrontierCandidate] = []
        for cluster in clusters:
            cluster_unknown = set().union(*(unknown_neighbors_by_frontier.get(cell, set()) for cell in cluster))
            if len(cluster_unknown) < 2:
                continue
            nav_cell = min(
                cluster,
                key=lambda cell: _grid_distance_cells(cell, self.current_cell),
            )
            centroid_cell = _centroid_cell(cluster)
            room_hint = self.scenario.room_for_cell(nav_cell)
            evidence = [
                f"{len(cluster_unknown)} unknown neighbor cells",
                f"cluster size {len(cluster)}",
            ]
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
        updated: list[FrontierRecord] = []
        for record in self.frontier_memory.candidate_records():
            preview_goal = Nav2GoalRequest(
                goal_id=f"preview_{self.decision_index:03d}_{record.frontier_id}",
                goal_type="frontier_preview",
                target_pose=record.nav_pose,
                planner_id=self.config.nav2_planner_id,
                controller_id=self.config.nav2_controller_id,
                behavior_tree=self.config.nav2_behavior_tree,
                metadata={"frontier_id": record.frontier_id},
            )
            plan = self.nav2.compute_path(preview_goal, record=False)
            record.path_cost_m = plan.path_length_m if plan.status == "succeeded" else None
            updated.append(record)
        updated.sort(
            key=lambda record: (
                record.path_cost_m is None,
                record.path_cost_m if record.path_cost_m is not None else 1e9,
                -record.unknown_gain,
            )
        )
        return updated

    def _build_prompt_payload(self, candidate_records: list[FrontierRecord]) -> dict[str, Any]:
        recent_views = self.keyframes[-3:]
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
        frontier_memory_snapshot = self.frontier_memory.snapshot()
        return {
            "mission": (
                "Explore the apartment until the accessible map is complete, keep frontier memory stable, "
                "and prefer fast coverage growth over redundant local motion."
            ),
            "robot": {
                "pose": self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict(),
                "room_id": self.scenario.room_for_cell(self.current_cell),
                "coverage": round(self._coverage(), 3),
                "trajectory_points": len(self.trajectory),
                "sensor_range_m": self.config.sensor_range_m,
            },
            "frontier_memory": frontier_memory_snapshot,
            "candidate_frontiers": [record.to_dict() for record in candidate_records],
            "explored_areas": explored_areas,
            "recent_views": recent_views,
            "guardrails": {
                "finish_requires_frontier_exhaustion": True,
                "finish_coverage_threshold": self.config.finish_coverage_threshold,
                "navigation_must_use_nav2_goal_validation": True,
                "frontier_ids_must_come_from_prompt": True,
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
                    row.append("V" if frontier_cells[cell] in {"completed", "failed"} else "F")
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

    def _apply_finish_guardrail(self, decision: ExplorationDecision, candidate_records: list[FrontierRecord]) -> ExplorationDecision:
        reachable = [record for record in candidate_records if record.path_cost_m is not None]
        if decision.decision_type == "finish" and reachable and self._coverage() < self.config.finish_coverage_threshold:
            fallback = self.policy._heuristic_decision(  # intentional guardrail fallback
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

    def _log_policy_step(self, prompt_payload: dict[str, Any], decision: ExplorationDecision, trace: dict[str, Any]) -> None:
        if self.config.trace_policy_stdout:
            print(
                f"[exploration-policy] step={self.decision_index} coverage={self._coverage():.3f} "
                f"decision={decision.decision_type} frontier={decision.selected_frontier_id}"
            )
        if self.config.trace_llm_stdout:
            print("[exploration-llm] prompt")
            print(trace.get("prompt", ""))
            print("[exploration-llm] response")
            print(json.dumps(trace.get("response", {}), indent=2, sort_keys=True))
            if trace.get("llm_trace", {}).get("error"):
                print(f"[exploration-llm] error={trace['llm_trace']['error']}")
        self.decision_log.append(
            {
                "step_index": self.decision_index,
                "coverage": round(self._coverage(), 3),
                "decision": decision.to_dict(),
                "trace": trace,
                "candidate_frontier_ids": [item.get("frontier_id") for item in prompt_payload.get("candidate_frontiers", [])],
            }
        )

    def _push_progress_update(self, *, message: str, frontier_id: str | None) -> None:
        coverage = self._coverage()
        result = {
            "coverage": round(coverage, 3),
            "trajectory": self.trajectory[-12:],
            "keyframes": self.keyframes[-4:],
            "frontier_memory": self.frontier_memory.snapshot(),
            "active_frontier_id": frontier_id,
        }
        self.backend.update_external_task(
            self.task_id,
            progress=min(coverage, 0.98),
            message=message,
            result=result,
        )

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

        regions = []
        semantic_area_candidates: list[dict[str, Any]] = []
        for room_id, room in self.scenario.rooms.items():
            coverage = self._room_coverage(room_id)
            if coverage <= 0.12 and not self.room_objects_seen.get(room_id):
                continue
            evidence = [f"{item} visible" for item in sorted(self.room_objects_seen.get(room_id, set()))]
            if not evidence:
                evidence = ["geometry observed", "frontier traversal reached this room"]
            confidence = min(0.55 + coverage * 0.35 + min(len(evidence), 4) * 0.03, 0.98)
            regions.append(
                {
                    "region_id": room.region_id,
                    "label": room.label,
                    "confidence": round(confidence, 3),
                    "polygon_2d": [[float(x), float(y)] for x, y in room.polygon_2d],
                    "centroid": {"x": room.center_pose.x, "y": room.center_pose.y},
                    "adjacency": list(room.adjacency),
                    "representative_keyframes": self.room_frames.get(room_id, [])[:3],
                    "evidence": evidence[:8],
                    "default_waypoints": [
                        {
                            "name": f"{room.label}_center",
                            **room.center_pose.to_dict(),
                        },
                        {
                            "name": f"{room.label}_door",
                            **room.entry_pose.to_dict(),
                        },
                    ],
                }
            )
            for sub_area in room.sub_areas:
                if all(token.split()[0] in " ".join(evidence).lower() for token in sub_area.evidence):
                    semantic_area_candidates.append(
                        {
                            "area_id": sub_area.area_id,
                            "label": sub_area.label,
                            "parent_region_id": room.region_id,
                            "polygon_2d": [[float(x), float(y)] for x, y in sub_area.polygon_2d],
                            "confidence": 0.74,
                            "evidence": list(sub_area.evidence),
                        }
                    )

        aggregated_semantics = _aggregate_semantic_updates(self.semantic_updates)
        semantic_area_candidates.extend(aggregated_semantics)

        summary = (
            f"Agentic apartment exploration completed with coverage {self._coverage():.3f}, "
            f"{len(regions)} mapped regions, and {len(self.decision_log)} exploration decisions."
        )
        return {
            "map_id": self.config.session,
            "frame": "map",
            "resolution": float(self.config.occupancy_resolution),
            "coverage": round(self._coverage(), 3),
            "summary": summary,
            "approved": False,
            "created_at": time.time(),
            "source": self.config.source,
            "mode": "sim_agentic",
            "trajectory": self.trajectory,
            "keyframes": self.keyframes,
            "regions": regions,
            "named_places": [],
            "occupancy": {
                "resolution": float(self.config.occupancy_resolution),
                "bounds": self.scenario.bounds(),
                "cells": occupancy_cells,
            },
            "frontiers": [record.to_dict() for record in self.frontier_memory.records.values()],
            "semantic_area_candidates": semantic_area_candidates,
            "artifacts": {
                "layout_id": self.scenario.layout_id,
                "decision_log": self.decision_log,
                "frontier_memory": self.frontier_memory.snapshot(),
                "guardrail_events": self.guardrail_events,
                "navigation": {
                    "control_steps": self.control_steps,
                    "total_distance_m": round(self.total_distance_m, 3),
                },
                "nav2": self.nav2.snapshot(),
                "llm_policy": {
                    "explorer_policy": self.config.explorer_policy,
                    "provider": self.config.llm_provider,
                    "model": self.config.llm_model,
                },
            },
        }

    def _sleep(self) -> None:
        if self.config.realtime_sleep_s > 0:
            time.sleep(self.config.realtime_sleep_s)

    def _make_nav2_goal(self, pose: Pose2D, *, goal_type: str, reason: str) -> Nav2GoalRequest:
        self.nav2_goal_counter += 1
        return Nav2GoalRequest(
            goal_id=f"nav2_goal_{self.nav2_goal_counter:03d}",
            goal_type=goal_type,
            target_pose=pose,
            planner_id=self.config.nav2_planner_id,
            controller_id=self.config.nav2_controller_id,
            behavior_tree=self.config.nav2_behavior_tree,
            metadata={"reason": reason},
        )

    def _on_nav2_motion_step(
        self,
        previous: GridCell,
        nxt: GridCell,
        goal: Nav2GoalRequest,
        step_index: int,
        total_waypoints: int,
    ) -> None:
        self.current_yaw = math.atan2(nxt.y - previous.y, nxt.x - previous.x)
        self.current_cell = nxt
        self.control_steps += 1
        self.total_distance_m += self.scenario.resolution
        self.trajectory.append(self.current_cell.center_pose(self.scenario.resolution, yaw=self.current_yaw).to_dict())
        capture_frame = self.config.show_cameras and self.control_steps % max(self.config.camera_log_stride, 1) == 0
        self._perform_scan(
            full_turnaround=False,
            capture_frame=capture_frame,
            reason=f"travel::{goal.goal_id}",
        )
        self._push_progress_update(
            message=(
                f"Nav2 executing {goal.goal_type} {goal.goal_id} "
                f"({step_index}/{max(total_waypoints - 1, 1)}). Coverage {self._coverage():.3f}."
            ),
            frontier_id=goal.metadata.get("reason"),
        )

    def _on_nav2_runtime_obstacle(self, cell: GridCell) -> None:
        self.known_cells[cell] = "occupied"


class ManiSkillExplorationRunner:
    def __init__(self, config: SimExplorationConfig, backend: ExplorationBackend) -> None:
        self.config = config
        self.backend = backend

    def run(self) -> dict[str, Any]:
        task = self.backend.begin_external_task(
            tool_id="explore",
            area=self.config.area,
            session=self.config.session,
            source=self.config.source,
            message="Starting agentic apartment exploration run.",
        )
        session = _ApartmentExplorationSession(self.config, self.backend, str(task["task_id"]))
        try:
            map_payload = session.run()
        except Exception as exc:
            self.backend.fail_external_task(
                str(task["task_id"]),
                message=f"Exploration backend failed: {exc}",
                result={"error": str(exc)},
            )
            raise

        self.backend.complete_external_task(
            str(task["task_id"]),
            map_payload=map_payload,
            message=(
                f"Completed apartment exploration with coverage {map_payload['coverage']:.3f} "
                f"and {len(map_payload['regions'])} semantic region(s)."
            ),
            result={
                "map": map_payload,
                "coverage": map_payload["coverage"],
                "region_count": len(map_payload["regions"]),
                "decision_count": len(map_payload["artifacts"]["decision_log"]),
            },
        )
        return self.backend.snapshot()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the simulated XLeRobot apartment exploration backend with frontier memory, "
            "occupancy mapping, and an optional LLM frontier-selection policy."
        )
    )
    parser.add_argument("--repo-root", default=str(Path.home() / "XLeRobot"))
    parser.add_argument("--persist-path", default="./artifacts/xlerobot_exploration_map.json")
    parser.add_argument("--area", default="apartment")
    parser.add_argument("--session", default="house_v1")
    parser.add_argument("--source", default="operator")
    parser.add_argument("--env-id", default="SceneManipulation-v1")
    parser.add_argument("--robot-uid", default="xlerobot")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos_dual_arm")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--occupancy-resolution", type=float, default=0.25)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rerun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-log-stride", type=int, default=2)
    parser.add_argument("--realtime-sleep-s", type=float, default=0.01)
    parser.add_argument("--explorer-policy", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--llm-model", default="mock")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-reasoning-effort", default=None)
    parser.add_argument("--trace-policy-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-llm-stdout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--serve-review-ui", action="store_true")
    parser.add_argument("--review-host", default="127.0.0.1")
    parser.add_argument("--review-port", type=int, default=8770)
    parser.add_argument("--sensor-range-m", type=float, default=10.0)
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--nav2-planner-id", default="GridBased")
    parser.add_argument("--nav2-controller-id", default="FollowPath")
    parser.add_argument("--nav2-behavior-tree", default="navigate_to_pose_w_replanning_and_recovery.xml")
    parser.add_argument("--nav2-recovery-enabled", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = ExplorationBackend(
        ExplorationBackendConfig(
            mode="sim",
            persist_path=args.persist_path,
            occupancy_resolution=args.occupancy_resolution,
        )
    )
    runner = ManiSkillExplorationRunner(
        SimExplorationConfig(
            repo_root=args.repo_root,
            persist_path=args.persist_path,
            env_id=args.env_id,
            robot_uid=args.robot_uid,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            shader=args.shader,
            sim_backend=args.sim_backend,
            num_envs=args.num_envs,
            force_reload=args.force_reload,
            area=args.area,
            session=args.session,
            source=args.source,
            occupancy_resolution=args.occupancy_resolution,
            max_control_steps=args.max_control_steps,
            max_episode_steps=args.max_episode_steps,
            show_cameras=args.show_cameras,
            use_rerun=args.use_rerun,
            camera_log_stride=args.camera_log_stride,
            realtime_sleep_s=args.realtime_sleep_s,
            explorer_policy=args.explorer_policy,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_reasoning_effort=args.llm_reasoning_effort,
            trace_policy_stdout=args.trace_policy_stdout,
            trace_llm_stdout=args.trace_llm_stdout,
            serve_review_ui=args.serve_review_ui,
            review_host=args.review_host,
            review_port=args.review_port,
            sensor_range_m=args.sensor_range_m,
            finish_coverage_threshold=args.finish_coverage_threshold,
            max_decisions=args.max_decisions,
            nav2_planner_id=args.nav2_planner_id,
            nav2_controller_id=args.nav2_controller_id,
            nav2_behavior_tree=args.nav2_behavior_tree,
            nav2_recovery_enabled=args.nav2_recovery_enabled,
        ),
        backend,
    )
    snapshot = runner.run()
    current_map = snapshot.get("current_map") or {}
    print(
        f"Saved exploration map `{current_map.get('map_id', args.session)}` "
        f"with {len(current_map.get('regions', []))} region(s), "
        f"coverage {current_map.get('coverage', 0.0)}, to {args.persist_path}."
    )
    if args.serve_review_ui:
        controller = LocalExplorationUIController(backend)
        server = ExplorationReviewServer(
            controller,
            host=args.review_host,
            port=args.review_port,
            allow_task_controls=False,
        )
        print(f"XLeRobot exploration review UI: http://{args.review_host}:{args.review_port}")
        server.serve_forever()
    return 0


def _build_simple_apartment(resolution: float) -> ApartmentScenario:
    width_m = 11.0
    height_m = 8.5
    width_cells = int(math.ceil(width_m / resolution))
    height_cells = int(math.ceil(height_m / resolution))
    free_cells: set[GridCell] = set()
    room_by_cell: dict[GridCell, str] = {}

    def carve(room_id: str, bounds: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = bounds
        for cell in _rect_cells(bounds, resolution, width_cells, height_cells):
            free_cells.add(cell)
            room_by_cell[cell] = room_id

    room_bounds = {
        "region_hallway": (4.25, 1.0, 6.25, 7.5),
        "region_living_room": (0.75, 0.75, 4.25, 3.75),
        "region_bathroom": (0.75, 4.5, 3.25, 7.5),
        "region_kitchen": (6.25, 4.5, 10.25, 7.5),
        "region_bedroom": (6.25, 0.75, 10.25, 3.75),
    }
    door_bounds = {
        "region_hallway": [
            (4.25, 2.0, 4.75, 2.75),
            (3.25, 5.5, 4.25, 6.25),
            (6.25, 5.5, 6.75, 6.25),
            (6.25, 2.0, 6.75, 2.75),
        ]
    }
    for room_id, bounds in room_bounds.items():
        carve(room_id, bounds)
    for bounds in door_bounds["region_hallway"]:
        carve("region_hallway", bounds)

    obstacle_cells: set[GridCell] = set()
    for bounds in (
        (2.0, 1.75, 2.75, 2.5),
        (7.75, 5.5, 8.75, 6.25),
        (7.25, 1.25, 9.5, 2.75),
        (1.25, 5.0, 2.5, 6.25),
    ):
        for cell in _rect_cells(bounds, resolution, width_cells, height_cells):
            if cell in free_cells:
                obstacle_cells.add(cell)
                free_cells.discard(cell)
                room_by_cell.pop(cell, None)

    rooms = {
        "region_hallway": RoomSpec(
            region_id="region_hallway",
            label="hallway",
            polygon_2d=_polygon_from_bounds(room_bounds["region_hallway"]),
            adjacency=("region_living_room", "region_bathroom", "region_kitchen", "region_bedroom"),
            objects=("charging dock", "shoe rack", "wall art"),
            descriptions=("a central hallway that connects the apartment rooms",),
            entry_pose=Pose2D(5.25, 3.0, 0.0),
            center_pose=Pose2D(5.25, 4.25, 0.0),
        ),
        "region_living_room": RoomSpec(
            region_id="region_living_room",
            label="living_room",
            polygon_2d=_polygon_from_bounds(room_bounds["region_living_room"]),
            adjacency=("region_hallway",),
            objects=("sofa", "tv", "coffee table", "desk", "monitor", "chair"),
            descriptions=("a living room with a sofa, television wall, and a small desk corner",),
            entry_pose=Pose2D(4.5, 2.25, 0.0),
            center_pose=Pose2D(2.5, 2.25, 0.0),
            sub_areas=(
                SubAreaSpec(
                    area_id="subarea_living_room_office",
                    label="office_area",
                    polygon_2d=_polygon_from_bounds((0.75, 0.75, 2.25, 1.75)),
                    evidence=("desk visible", "monitor visible", "chair visible"),
                ),
            ),
        ),
        "region_bathroom": RoomSpec(
            region_id="region_bathroom",
            label="bathroom",
            polygon_2d=_polygon_from_bounds(room_bounds["region_bathroom"]),
            adjacency=("region_hallway",),
            objects=("sink", "mirror", "toilet", "bathtub"),
            descriptions=("a compact bathroom with a sink, mirror, toilet, and bathtub",),
            entry_pose=Pose2D(3.75, 5.75, 0.0),
            center_pose=Pose2D(2.0, 6.0, 0.0),
        ),
        "region_kitchen": RoomSpec(
            region_id="region_kitchen",
            label="kitchen",
            polygon_2d=_polygon_from_bounds(room_bounds["region_kitchen"]),
            adjacency=("region_hallway",),
            objects=("fridge", "sink", "oven", "counter", "island"),
            descriptions=("a kitchen with a fridge, oven, sink, counter, and central island",),
            entry_pose=Pose2D(6.5, 5.75, 0.0),
            center_pose=Pose2D(8.25, 6.0, 0.0),
        ),
        "region_bedroom": RoomSpec(
            region_id="region_bedroom",
            label="bedroom",
            polygon_2d=_polygon_from_bounds(room_bounds["region_bedroom"]),
            adjacency=("region_hallway",),
            objects=("bed", "wardrobe", "lamp", "nightstand"),
            descriptions=("a bedroom with a bed, wardrobe, and bedside furniture",),
            entry_pose=Pose2D(6.5, 2.25, 0.0),
            center_pose=Pose2D(8.25, 2.25, 0.0),
        ),
    }

    objects = [
        WorldObject("obj_dock", "charging dock", "region_hallway", GridCell(int(4.75 / resolution), int(6.5 / resolution)), "robot charging dock"),
        WorldObject("obj_shoes", "shoe rack", "region_hallway", GridCell(int(5.5 / resolution), int(1.5 / resolution)), "shoe rack by the wall"),
        WorldObject("obj_sofa", "sofa", "region_living_room", GridCell(int(1.5 / resolution), int(2.75 / resolution)), "large sofa"),
        WorldObject("obj_tv", "tv", "region_living_room", GridCell(int(3.5 / resolution), int(3.0 / resolution)), "television on the far wall"),
        WorldObject("obj_coffee", "coffee table", "region_living_room", GridCell(int(2.25 / resolution), int(1.5 / resolution)), "low coffee table"),
        WorldObject("obj_desk", "desk", "region_living_room", GridCell(int(1.25 / resolution), int(1.25 / resolution)), "small office desk"),
        WorldObject("obj_monitor", "monitor", "region_living_room", GridCell(int(1.5 / resolution), int(1.0 / resolution)), "computer monitor"),
        WorldObject("obj_chair", "chair", "region_living_room", GridCell(int(1.75 / resolution), int(1.25 / resolution)), "desk chair"),
        WorldObject("obj_sink_bath", "sink", "region_bathroom", GridCell(int(2.75 / resolution), int(6.75 / resolution)), "bathroom sink"),
        WorldObject("obj_mirror", "mirror", "region_bathroom", GridCell(int(2.75 / resolution), int(7.0 / resolution)), "bathroom mirror"),
        WorldObject("obj_toilet", "toilet", "region_bathroom", GridCell(int(2.5 / resolution), int(5.5 / resolution)), "toilet"),
        WorldObject("obj_tub", "bathtub", "region_bathroom", GridCell(int(1.25 / resolution), int(5.5 / resolution)), "bathtub"),
        WorldObject("obj_fridge", "fridge", "region_kitchen", GridCell(int(9.5 / resolution), int(6.75 / resolution)), "kitchen fridge"),
        WorldObject("obj_sink_kitchen", "sink", "region_kitchen", GridCell(int(9.0 / resolution), int(5.0 / resolution)), "kitchen sink"),
        WorldObject("obj_oven", "oven", "region_kitchen", GridCell(int(7.0 / resolution), int(4.75 / resolution)), "kitchen oven"),
        WorldObject("obj_counter", "counter", "region_kitchen", GridCell(int(8.0 / resolution), int(4.75 / resolution)), "counter workspace"),
        WorldObject("obj_island", "island", "region_kitchen", GridCell(int(8.25 / resolution), int(6.0 / resolution)), "kitchen island"),
        WorldObject("obj_bed", "bed", "region_bedroom", GridCell(int(8.25 / resolution), int(2.25 / resolution)), "bed"),
        WorldObject("obj_wardrobe", "wardrobe", "region_bedroom", GridCell(int(9.5 / resolution), int(3.0 / resolution)), "wardrobe"),
        WorldObject("obj_lamp", "lamp", "region_bedroom", GridCell(int(6.75 / resolution), int(3.0 / resolution)), "floor lamp"),
        WorldObject("obj_nightstand", "nightstand", "region_bedroom", GridCell(int(7.0 / resolution), int(1.25 / resolution)), "nightstand"),
    ]

    free_cells -= obstacle_cells
    start_cell = GridCell(int(5.25 / resolution), int(4.25 / resolution))
    return ApartmentScenario(
        scenario_id="scenario_simple_apartment",
        layout_id="simple_apartment",
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        start_cell=start_cell,
        free_cells=free_cells,
        obstacle_cells=obstacle_cells,
        room_by_cell=room_by_cell,
        rooms=rooms,
        objects=objects,
    )


def _simulate_scan(
    scenario: ApartmentScenario,
    origin_cell: GridCell,
    *,
    yaw: float,
    max_range_m: float,
    full_turnaround: bool,
) -> ScanResult:
    origin_pose = origin_cell.center_pose(scenario.resolution, yaw=yaw)
    angles = _scan_angles(yaw, full_turnaround=full_turnaround)
    observed_free: set[GridCell] = set()
    observed_occupied: set[GridCell] = set()
    range_edge_frontiers: set[GridCell] = set()
    depth_hits: list[float] = []
    substep_m = scenario.resolution * 0.5
    max_steps = max(1, int(max_range_m / substep_m))

    for angle in angles:
        last_free: GridCell | None = None
        hit_obstacle = False
        seen_cells: set[GridCell] = set()
        for step in range(1, max_steps + 1):
            distance_m = step * substep_m
            x = origin_pose.x + math.cos(angle) * distance_m
            y = origin_pose.y + math.sin(angle) * distance_m
            cell = scenario.world_to_cell(x, y)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            if not scenario.in_bounds(cell):
                break
            if scenario.is_occupied(cell):
                observed_occupied.add(cell)
                depth_hits.append(distance_m)
                hit_obstacle = True
                break
            observed_free.add(cell)
            last_free = cell
        if not hit_obstacle and last_free is not None:
            range_edge_frontiers.add(last_free)

    visible_objects: list[str] = []
    visible_room_ids: set[str] = set()
    for cell in observed_free:
        room_id = scenario.room_for_cell(cell)
        if room_id:
            visible_room_ids.add(room_id)

    for item in scenario.objects:
        if item.cell == origin_cell:
            visible_objects.append(item.label)
            continue
        distance_m = _grid_distance_cells(origin_cell, item.cell) * scenario.resolution
        if distance_m > max_range_m:
            continue
        if not full_turnaround and _angle_difference(yaw, math.atan2(item.cell.y - origin_cell.y, item.cell.x - origin_cell.x)) > math.radians(60):
            continue
        if _line_of_sight_clear(scenario, origin_cell, item.cell):
            visible_objects.append(item.label)
            depth_hits.append(distance_m)
            visible_room_ids.add(item.room_id)

    visible_objects = sorted(set(visible_objects))
    current_room_id = scenario.room_for_cell(origin_cell)
    room_label = scenario.rooms.get(current_room_id).label if current_room_id in scenario.rooms else "unknown_space"
    description = (
        f"{'360 degree' if full_turnaround else 'forward'} scan from {room_label}. "
        f"Visible objects: {', '.join(visible_objects[:6]) or 'none'}."
    )
    thumbnail = _thumbnail_data_url(
        title=room_label.replace("_", " ").title(),
        subtitle=", ".join(visible_objects[:3]) or "mapping scan",
    )
    return ScanResult(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        range_edge_frontiers=range_edge_frontiers,
        visible_objects=visible_objects,
        visible_room_ids=sorted(visible_room_ids),
        point_count=max((len(observed_free) + len(observed_occupied)) * 18, 900),
        depth_min_m=round(min(depth_hits), 3) if depth_hits else 0.35,
        depth_max_m=round(max(depth_hits), 3) if depth_hits else max_range_m,
        description=description,
        thumbnail_data_url=thumbnail,
    )


def _scan_angles(yaw: float, *, full_turnaround: bool) -> list[float]:
    if full_turnaround:
        return [math.radians(index * 5.0) for index in range(72)]
    return [yaw + math.radians(offset) for offset in range(-55, 56, 5)]


def _rect_cells(bounds: tuple[float, float, float, float], resolution: float, width_cells: int, height_cells: int) -> set[GridCell]:
    x0, y0, x1, y1 = bounds
    cells: set[GridCell] = set()
    for x in range(width_cells):
        for y in range(height_cells):
            center_x = (x + 0.5) * resolution
            center_y = (y + 0.5) * resolution
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                cells.add(GridCell(x, y))
    return cells


def _polygon_from_bounds(bounds: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = bounds
    return (
        (round(x0, 3), round(y0, 3)),
        (round(x1, 3), round(y0, 3)),
        (round(x1, 3), round(y1, 3)),
        (round(x0, 3), round(y1, 3)),
    )


def _neighbors4(cell: GridCell) -> tuple[GridCell, ...]:
    return (
        GridCell(cell.x + 1, cell.y),
        GridCell(cell.x - 1, cell.y),
        GridCell(cell.x, cell.y + 1),
        GridCell(cell.x, cell.y - 1),
    )


def _neighbors8(cell: GridCell) -> tuple[GridCell, ...]:
    return tuple(
        GridCell(cell.x + dx, cell.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    )


def _grid_distance_cells(a: GridCell, b: GridCell) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _centroid_cell(cells: Iterable[GridCell]) -> GridCell:
    normalized = list(cells)
    if not normalized:
        return GridCell(0, 0)
    mean_x = round(sum(cell.x for cell in normalized) / len(normalized))
    mean_y = round(sum(cell.y for cell in normalized) / len(normalized))
    return GridCell(mean_x, mean_y)


def _search_known_safe_path(start: GridCell, goal: GridCell, traversable: set[GridCell]) -> list[GridCell]:
    if start == goal:
        return [start]
    if start not in traversable or goal not in traversable:
        return []
    open_heap: list[tuple[int, int, GridCell]] = []
    heapq.heappush(open_heap, (_grid_distance_cells(start, goal), 0, start))
    came_from: dict[GridCell, GridCell] = {}
    g_score: dict[GridCell, int] = {start: 0}
    seen: set[GridCell] = set()
    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current in seen:
            continue
        seen.add(current)
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _neighbors4(current):
            if neighbor not in traversable:
                continue
            tentative = cost + 1
            if tentative < g_score.get(neighbor, 1_000_000):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + _grid_distance_cells(neighbor, goal), tentative, neighbor))
    return []


def _reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _line_of_sight_clear(scenario: ApartmentScenario, start: GridCell, goal: GridCell) -> bool:
    start_pose = start.center_pose(scenario.resolution)
    goal_pose = goal.center_pose(scenario.resolution)
    distance = math.hypot(goal_pose.x - start_pose.x, goal_pose.y - start_pose.y)
    if distance <= 1e-6:
        return True
    steps = max(1, int(distance / (scenario.resolution * 0.35)))
    seen: set[GridCell] = set()
    for step in range(1, steps):
        ratio = step / steps
        x = start_pose.x + (goal_pose.x - start_pose.x) * ratio
        y = start_pose.y + (goal_pose.y - start_pose.y) * ratio
        cell = scenario.world_to_cell(x, y)
        if cell in seen or cell == start or cell == goal:
            continue
        seen.add(cell)
        if scenario.is_occupied(cell):
            return False
    return True


def _angle_difference(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(delta)


def _pose_distance_m(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _thumbnail_data_url(*, title: str, subtitle: str) -> str:
    import base64

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#9a3412"/>
    </linearGradient>
  </defs>
  <rect width="320" height="180" rx="20" fill="#f8fafc"/>
  <rect x="14" y="14" width="292" height="152" rx="16" fill="url(#g)" opacity="0.16"/>
  <text x="28" y="64" font-family="IBM Plex Sans, Arial, sans-serif" font-size="28" fill="#102a43">{title}</text>
  <text x="28" y="98" font-family="IBM Plex Sans, Arial, sans-serif" font-size="14" fill="#486581">{subtitle}</text>
  <circle cx="250" cy="58" r="20" fill="#0f766e" opacity="0.2"/>
  <circle cx="218" cy="118" r="34" fill="#9a3412" opacity="0.12"/>
</svg>
""".strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def _dedupe_text(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        output.append(normalized)
        seen.add(normalized)
    return output


def _aggregate_semantic_updates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        label = str(item.get("label", "")).strip()
        target_id = str(item.get("target_id", "")).strip()
        if not label or not target_id:
            continue
        key = (label, target_id)
        existing = aggregated.get(key)
        confidence = float(item.get("confidence", 0.5))
        evidence = [str(token) for token in item.get("evidence", []) if str(token).strip()]
        candidate = {
            "area_id": f"semantic_{target_id}_{label}",
            "label": label,
            "kind": str(item.get("kind", "room_hint")),
            "target_id": target_id,
            "confidence": round(confidence, 3),
            "evidence": evidence[:4],
        }
        if existing is None or candidate["confidence"] > existing["confidence"]:
            aggregated[key] = candidate
    return list(aggregated.values())


if __name__ == "__main__":
    raise SystemExit(main())
