# Testing Nav2 Bridge Connection

This file collects the terminal commands for testing the ManiSkill to ROS 2 bridge, `slam_toolbox`, Nav2, and the goal client together.

## Terminal 1: bridge

```bash
cd /home/alin/Robot42
conda deactivate || true
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python scripts/render_xlerobot_nav2_params.py --ros-distro humble

python examples/xlerobot_nav2_bridge_playground.py \
  --render-mode human \
  --env-id SceneManipulation-v1 \
  --build-config-idx 0 \
  --realtime-factor 0 \
  --no-publish-head-camera \
  --max-episode-steps 1000
```

## Terminal 2: slam_toolbox

```bash
conda deactivate || true
source /opt/ros/humble/setup.bash

ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=true \
  slam_params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
```

## Terminal 3: Nav2

```bash
conda deactivate || true
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

## Terminal 4: verify the graph

```bash
conda deactivate || true
source /opt/ros/humble/setup.bash

ros2 node list
ros2 topic info /map
ros2 topic echo /odom --once
ros2 topic echo /tf --once
ros2 action info /compute_path_to_pose
ros2 action info /navigate_to_pose
```

Expected checks:

- `/xlerobot_maniskill_ros_bridge` is present
- `/slam_toolbox` is present
- `/map` has `Publisher count: 1`
- `/compute_path_to_pose` has `Action servers: 1`
- `/navigate_to_pose` has `Action servers: 1`

## Terminal 4: send a small goal

```bash
conda deactivate || true
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.nav2_goal_client \
  --x 0.3 \
  --y 0.0 \
  --yaw 0.0 \
  --compute-path-first
```

If `ComputePathToPose` returns `0 poses`, do not interpret that as a forward-speed issue. It means Nav2 does not yet have a valid path in the current map. In practice, let the robot build more local map coverage first, ideally with an initial 360 degree scan before relying on forward goals.

## Slightly larger goal

```bash
conda deactivate || true
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.nav2_goal_client \
  --x 0.8 \
  --y 0.0 \
  --yaw 0.0 \
  --compute-path-first
```

## Duplicate Nav2 cleanup

If the goal client reports more than one action server for `compute_path_to_pose` or `navigate_to_pose`, stop stale Nav2 stacks before retrying:

```bash
ps -ef | grep -E 'nav2_bringup|navigation_launch.py|planner_server|controller_server|bt_navigator|behavior_server|lifecycle_manager_navigation' | grep -v grep
pkill -f 'ros2 launch nav2_bringup navigation_launch.py' || true
source /opt/ros/humble/setup.bash
ros2 daemon stop
ros2 daemon start
ros2 node list
ros2 action info /compute_path_to_pose
```

Then restart only Terminal 3.

## Speed retuning

The current generated defaults are:

- forward max velocity: `0.65`
- angular max velocity: `0.45`
- DWB rotation bias reduced:
  `RotateToGoal.scale = 8.0`
  `GoalAlign.scale = 12.0`
  `PathAlign.scale = 16.0`

If you change these defaults in code, rerender the Nav2 params and restart Terminal 3.
