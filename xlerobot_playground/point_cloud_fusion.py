from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np

from xlerobot_playground.map_editing import OccupancyFusionConfig, merge_occupancy_observation


POINT_CLOUD_OCCUPANCY_FUSION_CONFIG = OccupancyFusionConfig(
    occupied_observation_weight=1.0,
    free_observation_weight=-0.20,
    occupied_enter_threshold=2.0,
    occupied_exit_threshold=0.10,
    min_score=-3.0,
    max_score=6.0,
)


@dataclass(frozen=True)
class PointCloudFusionConfig:
    range_min_m: float = 0.25
    range_max_m: float = 4.0
    floor_free_min_z_m: float = -0.08
    floor_free_max_z_m: float = 0.08
    obstacle_min_z_m: float = 0.08
    robot_clearance_height_m: float = 1.50
    obstacle_max_z_m: float = 1.80
    free_ray_max_m: float = 4.0
    voxel_size_m: float | None = None
    min_points_per_occupied_cell: int = 2
    obstacle_inflation_radius_m: float = 0.10
    max_rays: int = 2400
    evidence: OccupancyFusionConfig = POINT_CLOUD_OCCUPANCY_FUSION_CONFIG


@dataclass(frozen=True)
class PointCloudIntegrationSummary:
    raw_point_count: int
    valid_point_count: int
    voxel_point_count: int
    floor_point_count: int
    obstacle_point_count: int
    occupied_cell_count: int
    free_cell_count: int
    invalid_point_count: int
    integrated_rays: int


def integrate_transformed_point_cloud_observation(
    *,
    sensor_origin_xyz: Sequence[float],
    points_xyz_map: Any,
    map_resolution_m: float,
    cell_from_world: Callable[[float, float], Any],
    known_cells: dict[Any, str] | None,
    evidence_scores: dict[Any, float] | None,
    range_edge_cells: set[Any] | None = None,
    visited_cells: set[Any] | None = None,
    config: PointCloudFusionConfig = PointCloudFusionConfig(),
) -> PointCloudIntegrationSummary:
    points = np.asarray(points_xyz_map, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        points = np.empty((0, 3), dtype=np.float32)
    else:
        points = points[:, :3]
    origin = np.asarray(sensor_origin_xyz, dtype=np.float32).reshape(3)
    raw_count = int(points.shape[0])
    if raw_count <= 0:
        return PointCloudIntegrationSummary(0, 0, 0, 0, 0, 0, 0, 0, 0)

    vectors = points - origin.reshape(1, 3)
    ranges = np.linalg.norm(vectors, axis=1)
    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(ranges)
    range_mask = (ranges >= float(config.range_min_m)) & (ranges <= float(config.range_max_m))
    valid_mask = finite_mask & range_mask
    valid_points = points[valid_mask]
    valid_ranges = ranges[valid_mask]
    invalid_count = raw_count - int(valid_points.shape[0])
    if valid_points.size == 0:
        return PointCloudIntegrationSummary(raw_count, 0, 0, 0, 0, 0, 0, invalid_count, 0)

    voxel_size = float(config.voxel_size_m or max(float(map_resolution_m) * 0.5, 0.03))
    valid_points, valid_ranges = _voxel_downsample(valid_points, valid_ranges, voxel_size)
    voxel_count = int(valid_points.shape[0])
    if voxel_count <= 0:
        return PointCloudIntegrationSummary(raw_count, 0, 0, 0, 0, 0, 0, invalid_count, 0)

    origin_cell = cell_from_world(float(origin[0]), float(origin[1]))
    if visited_cells is not None:
        visited_cells.add(origin_cell)
    if known_cells is not None:
        merge_occupancy_observation(
            known_cells,
            origin_cell,
            "free",
            evidence_scores=evidence_scores,
            config=config.evidence,
        )

    floor_mask = (
        (valid_points[:, 2] >= float(config.floor_free_min_z_m))
        & (valid_points[:, 2] <= float(config.floor_free_max_z_m))
    )
    obstacle_mask = (
        (valid_points[:, 2] >= float(config.obstacle_min_z_m))
        & (valid_points[:, 2] <= min(float(config.obstacle_max_z_m), float(config.robot_clearance_height_m)))
    )
    free_targets = valid_points[floor_mask | obstacle_mask]
    free_ranges = valid_ranges[floor_mask | obstacle_mask]
    obstacle_points = valid_points[obstacle_mask]

    occupied_observation_counts: dict[Any, int] = {}
    inflation_cells = max(0, int(math.ceil(float(config.obstacle_inflation_radius_m) / max(float(map_resolution_m), 1e-6))))
    for point in obstacle_points:
        center_cell = cell_from_world(float(point[0]), float(point[1]))
        for cell in _inflated_cells(center_cell, inflation_cells):
            occupied_observation_counts[cell] = occupied_observation_counts.get(cell, 0) + 1

    occupied_cells = {
        cell
        for cell, count in occupied_observation_counts.items()
        if count >= max(int(config.min_points_per_occupied_cell), 1)
    }
    if known_cells is not None:
        for cell in occupied_cells:
            for _ in range(max(occupied_observation_counts.get(cell, 1), 1)):
                merge_occupancy_observation(
                    known_cells,
                    cell,
                    "occupied",
                    evidence_scores=evidence_scores,
                    config=config.evidence,
                )

    free_cells: set[Any] = set()
    integrated_rays = 0
    if free_targets.size:
        ray_indices = _ray_sample_indices(int(free_targets.shape[0]), max(int(config.max_rays), 1))
        step_m = max(float(map_resolution_m) * 0.5, 0.04)
        for index in ray_indices:
            target = free_targets[index]
            ray_range = min(float(free_ranges[index]), float(config.free_ray_max_m))
            if ray_range <= float(config.range_min_m):
                continue
            target_cell = cell_from_world(float(target[0]), float(target[1]))
            is_obstacle_endpoint = target_cell in occupied_cells
            ray_cells = _ray_cells(
                origin=origin,
                target=target,
                ray_range_m=ray_range,
                step_m=step_m,
                cell_from_world=cell_from_world,
            )
            if is_obstacle_endpoint and ray_cells and ray_cells[-1] == target_cell:
                ray_cells = ray_cells[:-1]
            for cell in ray_cells:
                if visited_cells is not None:
                    visited_cells.add(cell)
                if cell in occupied_cells:
                    continue
                free_cells.add(cell)
            range_edge_threshold = min(float(config.range_max_m), float(config.free_ray_max_m)) * 0.98
            if (
                range_edge_cells is not None
                and not is_obstacle_endpoint
                and ray_cells
                and float(free_ranges[index]) >= range_edge_threshold
            ):
                range_edge_cells.add(ray_cells[-1])
            integrated_rays += 1

    if known_cells is not None:
        for cell in free_cells:
            merge_occupancy_observation(
                known_cells,
                cell,
                "free",
                evidence_scores=evidence_scores,
                config=config.evidence,
            )

    return PointCloudIntegrationSummary(
        raw_point_count=raw_count,
        valid_point_count=int(np.count_nonzero(valid_mask)),
        voxel_point_count=voxel_count,
        floor_point_count=int(np.count_nonzero(floor_mask)),
        obstacle_point_count=int(np.count_nonzero(obstacle_mask)),
        occupied_cell_count=len(occupied_cells),
        free_cell_count=len(free_cells),
        invalid_point_count=invalid_count,
        integrated_rays=integrated_rays,
    )


def _voxel_downsample(points: np.ndarray, ranges: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= 1:
        return points, ranges
    voxel_size = max(float(voxel_size_m), 1e-6)
    voxel_keys = np.floor(points / voxel_size).astype(np.int32)
    selected: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(voxel_keys):
        item = (int(key[0]), int(key[1]), int(key[2]))
        previous = selected.get(item)
        if previous is None or ranges[index] < ranges[previous]:
            selected[item] = index
    indices = np.fromiter(selected.values(), dtype=np.int64)
    return points[indices], ranges[indices]


def _inflated_cells(cell: Any, radius_cells: int) -> list[Any]:
    if radius_cells <= 0:
        return [cell]
    cells = []
    cell_type = type(cell)
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            cells.append(cell_type(cell.x + dx, cell.y + dy))
    return cells


def _ray_sample_indices(count: int, max_rays: int) -> range:
    if count <= max_rays:
        return range(count)
    stride = max(int(math.ceil(count / max_rays)), 1)
    return range(0, count, stride)


def _ray_cells(
    *,
    origin: np.ndarray,
    target: np.ndarray,
    ray_range_m: float,
    step_m: float,
    cell_from_world: Callable[[float, float], Any],
) -> list[Any]:
    vector = target - origin
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return []
    direction = vector / norm
    samples = max(1, int(math.ceil(float(ray_range_m) / max(float(step_m), 1e-6))))
    cells: list[Any] = []
    previous = None
    for index in range(1, samples + 1):
        distance = min(index * float(step_m), float(ray_range_m))
        point = origin + direction * distance
        cell = cell_from_world(float(point[0]), float(point[1]))
        if previous is None or cell != previous:
            cells.append(cell)
            previous = cell
    return cells
