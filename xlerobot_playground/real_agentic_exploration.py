from __future__ import annotations

import argparse

from xlerobot_playground.nav2_defaults import default_nav2_behavior_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live XLeRobot agentic exploration loop through ROS/Nav2. "
            "This expects robot_brain_agent, real_ros_bridge, RGB-D odometry, and Nav2 to already be running."
        )
    )
    parser.add_argument("--persist-path", default="./artifacts/real_xlerobot_exploration_map.json")
    parser.add_argument("--session", default="real_house_v1")
    parser.add_argument("--area", default="real_space")
    parser.add_argument("--source", default="real_xlerobot")
    parser.add_argument("--occupancy-resolution", type=float, default=0.10)
    parser.add_argument("--sensor-range-m", type=float, default=4.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.22)
    parser.add_argument("--frontier-min-opening-m", type=float, default=0.55)
    parser.add_argument("--visited-frontier-filter-radius-m", type=float, default=0.45)
    parser.add_argument("--finish-coverage-threshold", type=float, default=0.96)
    parser.add_argument("--max-decisions", type=int, default=32)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
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
    parser.add_argument("--serve-review-ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-host", default="0.0.0.0")
    parser.add_argument("--review-port", type=int, default=8770)
    parser.add_argument("--review-ui-flavor", choices=("user", "developer"), default="user")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wait-for-ui-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ros-navigation-map-source", choices=("fused_scan", "external"), default="fused_scan")
    parser.add_argument("--ros-map-topic", default="/map")
    parser.add_argument("--ros-scan-topic", default="/scan")
    parser.add_argument("--ros-rgb-topic", default="/camera/head/image_raw")
    parser.add_argument("--ros-cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-map-frame", default="map")
    parser.add_argument("--ros-odom-frame", default="odom")
    parser.add_argument("--ros-base-frame", default="base_link")
    parser.add_argument("--ros-server-timeout-s", type=float, default=10.0)
    parser.add_argument("--ros-ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--ros-turn-scan-timeout-s", type=float, default=75.0)
    parser.add_argument("--ros-turn-scan-settle-s", type=float, default=1.0)
    parser.add_argument("--ros-manual-spin-angular-speed-rad-s", type=float, default=0.10)
    parser.add_argument("--ros-manual-spin-publish-hz", type=float, default=10.0)
    parser.add_argument("--nav2-planner-id", default="GridBased")
    parser.add_argument("--nav2-controller-id", default="FollowPath")
    parser.add_argument("--nav2-behavior-tree", default=default_nav2_behavior_tree())
    parser.add_argument("--nav2-recovery-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ros-allow-multiple-action-servers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-waypoints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--automatic-semantic-waypoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-llm-provider", default=None)
    parser.add_argument("--semantic-llm-model", default=None)
    parser.add_argument("--semantic-llm-base-url", default=None)
    parser.add_argument("--semantic-llm-api-key", default=None)
    parser.add_argument("--semantic-vlm-async", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experimental-free-space-semantic-waypoints", action=argparse.BooleanOptionalAction, default=False)
    return parser


def translated_args(args: argparse.Namespace) -> list[str]:
    items = [
        "--persist-path",
        args.persist_path,
        "--area",
        args.area,
        "--session",
        args.session,
        "--source",
        args.source,
        "--occupancy-resolution",
        str(args.occupancy_resolution),
        "--sensor-range-m",
        str(args.sensor_range_m),
        "--robot-radius-m",
        str(args.robot_radius_m),
        "--frontier-min-opening-m",
        str(args.frontier_min_opening_m),
        "--visited-frontier-filter-radius-m",
        str(args.visited_frontier_filter_radius_m),
        "--finish-coverage-threshold",
        str(args.finish_coverage_threshold),
        "--max-decisions",
        str(args.max_decisions),
        "--nav2-mode",
        "ros",
        "--nav2-planner-id",
        args.nav2_planner_id,
        "--nav2-controller-id",
        args.nav2_controller_id,
        "--nav2-behavior-tree",
        args.nav2_behavior_tree,
        "--ros-navigation-map-source",
        args.ros_navigation_map_source,
        "--ros-map-topic",
        args.ros_map_topic,
        "--ros-scan-topic",
        args.ros_scan_topic,
        "--ros-rgb-topic",
        args.ros_rgb_topic,
        "--ros-cmd-vel-topic",
        args.ros_cmd_vel_topic,
        "--ros-map-frame",
        args.ros_map_frame,
        "--ros-odom-frame",
        args.ros_odom_frame,
        "--ros-base-frame",
        args.ros_base_frame,
        "--ros-server-timeout-s",
        str(args.ros_server_timeout_s),
        "--ros-ready-timeout-s",
        str(args.ros_ready_timeout_s),
        "--ros-turn-scan-timeout-s",
        str(args.ros_turn_scan_timeout_s),
        "--ros-turn-scan-settle-s",
        str(args.ros_turn_scan_settle_s),
        "--ros-manual-spin-angular-speed-rad-s",
        str(args.ros_manual_spin_angular_speed_rad_s),
        "--ros-manual-spin-publish-hz",
        str(args.ros_manual_spin_publish_hz),
        "--explorer-policy",
        args.explorer_policy,
        "--llm-provider",
        args.llm_provider,
        "--llm-model",
        args.llm_model,
        "--llm-temperature",
        str(args.llm_temperature),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--review-host",
        args.review_host,
        "--review-port",
        str(args.review_port),
        "--review-ui-flavor",
        args.review_ui_flavor,
    ]
    optional_pairs = [
        ("--max-control-steps", args.max_control_steps),
        ("--max-episode-steps", args.max_episode_steps),
        ("--llm-base-url", args.llm_base_url),
        ("--llm-api-key", args.llm_api_key),
        ("--llm-reasoning-effort", args.llm_reasoning_effort),
        ("--semantic-llm-provider", args.semantic_llm_provider),
        ("--semantic-llm-model", args.semantic_llm_model),
        ("--semantic-llm-base-url", args.semantic_llm_base_url),
        ("--semantic-llm-api-key", args.semantic_llm_api_key),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            items.extend([flag, str(value)])
    bool_flags = [
        ("--trace-policy-stdout", "--no-trace-policy-stdout", args.trace_policy_stdout),
        ("--trace-llm-stdout", "--no-trace-llm-stdout", args.trace_llm_stdout),
        ("--serve-review-ui", "--no-serve-review-ui", args.serve_review_ui),
        ("--open-browser", "--no-open-browser", args.open_browser),
        ("--wait-for-ui-start", "--no-wait-for-ui-start", args.wait_for_ui_start),
        ("--nav2-recovery-enabled", "--no-nav2-recovery-enabled", args.nav2_recovery_enabled),
        ("--ros-allow-multiple-action-servers", "--no-ros-allow-multiple-action-servers", args.ros_allow_multiple_action_servers),
        ("--semantic-waypoints-enabled", "--no-semantic-waypoints-enabled", args.semantic_waypoints_enabled),
        ("--automatic-semantic-waypoints", "--no-automatic-semantic-waypoints", args.automatic_semantic_waypoints),
        ("--semantic-vlm-async", "--no-semantic-vlm-async", args.semantic_vlm_async),
        (
            "--experimental-free-space-semantic-waypoints",
            "--no-experimental-free-space-semantic-waypoints",
            args.experimental_free_space_semantic_waypoints,
        ),
    ]
    for enabled_flag, disabled_flag, value in bool_flags:
        items.append(enabled_flag if value else disabled_flag)
    return items


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from xlerobot_playground.sim_exploration_backend import main as exploration_main

    return exploration_main(translated_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
