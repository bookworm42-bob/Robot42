from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import socket
import time
from typing import Any
from urllib import error
from urllib import request
from urllib.parse import urljoin


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True)
class SmokeStep:
    name: str
    translation_m: float = 0.0
    yaw_delta_rad: float = 0.0


def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def pose_from_payload(payload: Any) -> Pose2D:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected pose dict, got {type(payload).__name__}")
    return Pose2D(float(payload["x"]), float(payload["y"]), float(payload["yaw"]))


def relative_goal(start: Pose2D, step: SmokeStep) -> Pose2D:
    return Pose2D(
        x=start.x + step.translation_m * math.cos(start.yaw),
        y=start.y + step.translation_m * math.sin(start.yaw),
        yaw=angle_wrap(start.yaw + step.yaw_delta_rad),
    )


def distance_error_m(actual: Pose2D, goal: Pose2D) -> float:
    return math.hypot(actual.x - goal.x, actual.y - goal.y)


def yaw_error_rad(actual: Pose2D, goal: Pose2D) -> float:
    return abs(angle_wrap(actual.yaw - goal.yaw))


class HttpRequestError(RuntimeError):
    pass


def get_json(base_url: str, path: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    try:
        with request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except socket.timeout as exc:
        raise HttpRequestError(f"GET {url} timed out after {timeout_s:.1f}s") from exc
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpRequestError(f"GET {url} failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise HttpRequestError(f"GET {url} failed: {exc.reason}") from exc


def post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except socket.timeout as exc:
        raise HttpRequestError(f"POST {url} timed out after {timeout_s:.1f}s") from exc
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpRequestError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise HttpRequestError(f"POST {url} failed: {exc.reason}") from exc


class RouterClient:
    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        return get_json(self.base_url, "/api/health", timeout_s=min(self.timeout_s, 2.0))

    def current_pose(self) -> Pose2D:
        payload = get_json(self.base_url, "/api/router/current_pose", timeout_s=min(self.timeout_s, 2.0))
        return pose_from_payload(payload["pose"])

    def compute_path(self, goal: Pose2D, *, planner_id: str = "") -> list[Pose2D]:
        payload = post_json(
            self.base_url,
            "/api/router/compute_path_to_pose",
            {"goal_pose": goal.to_dict(), "planner_id": planner_id},
            timeout_s=max(self.timeout_s, 30.0),
        )
        status_label = str(payload.get("status_label", "unknown"))
        path = [pose_from_payload(item) for item in payload.get("path_poses", [])]
        if status_label != "succeeded" or not path:
            raise RuntimeError(f"Nav2 planner returned `{status_label}` with {len(path)} poses.")
        return path


class RobotBrainClient:
    def __init__(self, base_url: str, *, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        return get_json(self.base_url, "/health", timeout_s=self.timeout_s)

    def cmd_vel(self, *, linear_m_s: float, angular_rad_s: float) -> dict[str, Any]:
        return post_json(
            self.base_url,
            "/cmd_vel",
            {"linear_m_s": linear_m_s, "angular_rad_s": angular_rad_s},
            timeout_s=self.timeout_s,
        )

    def stop(self) -> dict[str, Any]:
        return post_json(self.base_url, "/stop", {}, timeout_s=self.timeout_s)


def velocity_command(
    pose: Pose2D,
    goal: Pose2D,
    *,
    max_linear_m_s: float,
    max_angular_rad_s: float,
    yaw_align_tolerance_rad: float,
    goal_distance_tolerance_m: float,
) -> tuple[float, float, bool]:
    distance = distance_error_m(pose, goal)
    final_yaw_error = angle_wrap(goal.yaw - pose.yaw)
    if distance <= goal_distance_tolerance_m:
        if abs(final_yaw_error) <= yaw_align_tolerance_rad:
            return 0.0, 0.0, True
        angular = max(-max_angular_rad_s, min(max_angular_rad_s, 1.5 * final_yaw_error))
        return 0.0, angular, False

    target_yaw = math.atan2(goal.y - pose.y, goal.x - pose.x)
    heading_error = angle_wrap(target_yaw - pose.yaw)
    if abs(heading_error) > yaw_align_tolerance_rad:
        angular = max(-max_angular_rad_s, min(max_angular_rad_s, 1.5 * heading_error))
        return 0.0, angular, False

    linear = min(max_linear_m_s, max(0.01, 0.6 * distance))
    angular = max(-max_angular_rad_s, min(max_angular_rad_s, 1.0 * heading_error))
    return linear, angular, False


def follow_goal(
    *,
    router: RouterClient,
    robot: RobotBrainClient,
    goal: Pose2D,
    timeout_s: float,
    publish_hz: float,
    max_linear_m_s: float,
    max_angular_rad_s: float,
    goal_distance_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> Pose2D:
    deadline = time.monotonic() + timeout_s
    sleep_s = 1.0 / max(publish_hz, 1e-6)
    try:
        while time.monotonic() < deadline:
            pose = router.current_pose()
            linear, angular, done = velocity_command(
                pose,
                goal,
                max_linear_m_s=max_linear_m_s,
                max_angular_rad_s=max_angular_rad_s,
                yaw_align_tolerance_rad=yaw_tolerance_rad,
                goal_distance_tolerance_m=goal_distance_tolerance_m,
            )
            if done:
                robot.stop()
                return pose
            robot.cmd_vel(linear_m_s=linear, angular_rad_s=angular)
            time.sleep(sleep_s)
        raise RuntimeError(f"Timed out following local path goal after {timeout_s:.1f}s.")
    finally:
        robot.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Robot-brain Mode-B smoke test: ask offload Nav2 router for paths, "
            "then follow them locally through robot_brain_agent velocity commands."
        )
    )
    parser.add_argument("--router-url", required=True, help="Offload Nav2 router URL, e.g. http://OFFLOAD_IP:8891")
    parser.add_argument("--robot-brain-url", default="http://127.0.0.1:8765")
    parser.add_argument("--planner-id", default="")
    parser.add_argument("--forward-m", type=float, default=0.05)
    parser.add_argument("--turn-deg", type=float, default=5.0)
    parser.add_argument("--max-linear-m-s", type=float, default=0.03)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.10)
    parser.add_argument("--goal-distance-tolerance-m", type=float, default=0.025)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=2.5)
    parser.add_argument("--max-translation-error-m", type=float, default=0.05)
    parser.add_argument("--max-yaw-error-deg", type=float, default=5.0)
    parser.add_argument("--step-timeout-s", type=float, default=15.0)
    parser.add_argument("--publish-hz", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check router, robot brain, and current pose connectivity without moving the robot.",
    )
    return parser


def steps_from_args(args: argparse.Namespace) -> tuple[SmokeStep, ...]:
    turn = math.radians(args.turn_deg)
    return (
        SmokeStep("forward", translation_m=float(args.forward_m)),
        SmokeStep("rotate_left", yaw_delta_rad=turn),
        SmokeStep("rotate_right", yaw_delta_rad=-turn),
    )


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    router = RouterClient(args.router_url, timeout_s=args.timeout_s)
    robot = RobotBrainClient(args.robot_brain_url, timeout_s=min(args.timeout_s, 3.0))
    router_health = router.health()
    robot_health = robot.health()
    if not robot_health.get("motion_enabled", False):
        raise RuntimeError("robot_brain_agent reports motion_enabled=false. Restart it with --allow-motion-commands.")
    current_pose = router.current_pose()

    if args.preflight_only:
        return {
            "ok": True,
            "mode": "robot_brain_follows_nav2_path",
            "preflight_only": True,
            "router_health": router_health,
            "robot_brain_health": robot_health,
            "current_pose": current_pose.to_dict(),
        }

    results: list[dict[str, Any]] = []
    yaw_tolerance_rad = math.radians(args.yaw_tolerance_deg)
    for step in steps_from_args(args):
        start = router.current_pose()
        goal = relative_goal(start, step)
        path = router.compute_path(goal, planner_id=args.planner_id)
        end = follow_goal(
            router=router,
            robot=robot,
            goal=goal,
            timeout_s=args.step_timeout_s,
            publish_hz=args.publish_hz,
            max_linear_m_s=args.max_linear_m_s,
            max_angular_rad_s=args.max_angular_rad_s,
            goal_distance_tolerance_m=args.goal_distance_tolerance_m,
            yaw_tolerance_rad=yaw_tolerance_rad,
        )
        translation_error = distance_error_m(end, goal)
        yaw_error = yaw_error_rad(end, goal)
        result = {
            "name": step.name,
            "start": start.to_dict(),
            "goal": goal.to_dict(),
            "end": end.to_dict(),
            "path_pose_count": len(path),
            "translation_error_m": round(translation_error, 4),
            "yaw_error_deg": round(math.degrees(yaw_error), 3),
        }
        results.append(result)
        if abs(step.translation_m) > 1e-6 and translation_error > args.max_translation_error_m:
            raise RuntimeError(f"{step.name} translation error too high: {translation_error:.3f}m")
        if abs(step.yaw_delta_rad) > 1e-6 and yaw_error > math.radians(args.max_yaw_error_deg):
            raise RuntimeError(f"{step.name} yaw error too high: {math.degrees(yaw_error):.2f}deg")

    return {
        "ok": True,
        "mode": "robot_brain_follows_nav2_path",
        "router_health": router_health,
        "robot_brain_health": robot_health,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke_test(args)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
