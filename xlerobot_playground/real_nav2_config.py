from __future__ import annotations

import argparse
import os
from pathlib import Path

def default_nav2_params_path() -> Path:
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    return Path(f"/opt/ros/{ros_distro}/share/nav2_bringup/params/nav2_params.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate real-XLeRobot SLAM Toolbox and Nav2 params.")
    parser.add_argument("--base-nav2-params", default=str(default_nav2_params_path()))
    parser.add_argument("--output-dir", default="artifacts/nav2")
    parser.add_argument("--slam-output", default="xlerobot_slam_toolbox.yaml")
    parser.add_argument("--nav2-output", default="xlerobot_nav2_params.yaml")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--global-map-topic", default="/projected_map")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-laser-range", type=float, default=6.0)
    parser.add_argument("--robot-length-m", type=float, default=0.3913)
    parser.add_argument("--robot-width-m", type=float, default=0.459)
    parser.add_argument("--max-linear-velocity", type=float, default=0.03)
    parser.add_argument("--max-angular-velocity", type=float, default=0.10)
    parser.add_argument("--local-costmap-width", type=int, default=2)
    parser.add_argument("--local-costmap-height", type=int, default=2)
    parser.add_argument("--transform-tolerance-s", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from xlerobot_playground.nav2_params import (
        dump_yaml,
        load_yaml,
        patch_nav2_params,
        rectangular_footprint,
        render_slam_toolbox_params,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slam_params = render_slam_toolbox_params(
        use_sim_time=False,
        scan_topic=args.scan_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        resolution=args.resolution,
        max_laser_range=args.max_laser_range,
    )
    slam_path = output_dir / args.slam_output
    dump_yaml(slam_path, slam_params)

    base_nav2 = load_yaml(args.base_nav2_params)
    footprint = rectangular_footprint(length_m=args.robot_length_m, width_m=args.robot_width_m)
    nav2_params = patch_nav2_params(
        base_nav2,
        use_sim_time=False,
        scan_topic=args.scan_topic,
        global_map_topic=args.global_map_topic,
        map_frame=args.map_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
        footprint=footprint,
        obstacle_max_range=args.max_laser_range * 0.95,
        raytrace_max_range=args.max_laser_range,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        local_costmap_width=args.local_costmap_width,
        local_costmap_height=args.local_costmap_height,
        transform_tolerance_s=args.transform_tolerance_s,
        inflation_radius=0.0,
    )
    nav2_path = output_dir / args.nav2_output
    dump_yaml(nav2_path, nav2_params)

    print(f"Wrote SLAM params: {slam_path}")
    print(f"Wrote Nav2 params: {nav2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
