# Running Exploration

This file collects the commands for running the XLeRobot exploration flow in three modes:

- teleport debug playground
- simulated runtime with the live review UI
- ROS 2 + Nav2 runtime with the live review UI

The unified launcher is:

```bash
python examples/xlerobot_exploration_playground.py
```

It supports:

- `--movement-mode teleport`
- `--movement-mode simulated`
- `--movement-mode ros`
- `--ui-flavor user`
- `--ui-flavor developer`

## Environment Activation

For ManiSkill-only runs:

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /home/alin/Robot42/.venv-maniskill/bin/activate
```

For ROS 2 + Nav2 runs:

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
```

## One-Time Nav2 Setup

If Nav2, `slam_toolbox`, or the ROS-enabled ManiSkill environment are not installed yet:

```bash
cd /home/alin/Robot42
./scripts/setup_xlerobot_maniskill_nav2_env.sh
```

Then generate the ROS/Nav2 parameter files:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
python scripts/render_xlerobot_nav2_params.py --ros-distro humble
```

That writes:

- `/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml`
- `/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml`

## Teleport Debug Playground

This is the developer-oriented step-through playground. It uses ManiSkill RGB-D, the shared frontier logic, the live web UI, and teleport movement instead of Nav2.

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode teleport \
  --ui-flavor developer \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434
```

Useful optional flags:

- `--spawn-facing front|left|right|back`
- `--scan-mode turnaround|front_only`
- `--review-port 8770`
- `--open-browser`

## Simulated Runtime With Review UI

This is the real exploration backend with the live review UI, but using the in-repo simulated navigation runtime instead of ROS/Nav2.

### User UI

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode simulated \
  --ui-flavor user \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434
```

### Developer UI

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode simulated \
  --ui-flavor developer \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --trace-policy-stdout
```

## ROS 2 + Nav2 Runtime

Use separate terminals.

### Terminal 1: ManiSkill to ROS 2 bridge

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python scripts/render_xlerobot_nav2_params.py --ros-distro humble

python examples/xlerobot_nav2_bridge_playground.py \
  --render-mode human \
  --env-id SceneManipulation-v1 \
  --build-config-idx 0 \
  --realtime-factor 0 \
  --linear-cmd-gain 1.5 \
  --angular-cmd-gain 0.32 \
  --no-publish-head-camera \
  --max-episode-steps 1000
```

### Terminal 2: `slam_toolbox`

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash

ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=true \
  slam_params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
```

### Terminal 3: Nav2

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

Run exactly one Nav2 launch at a time.

### Terminal 4: exploration runtime with real Nav2

### User UI

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode ros \
  --ui-flavor user \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434
```

### Developer UI

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode ros \
  --ui-flavor developer \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --trace-policy-stdout
```

## Manual Wall Editing

In both `user` and `developer` review UIs you can:

- pause exploration
- draw walls
- erase walls
- reset edited cells

The exploration loop halts while paused. This blocks further LLM decisions and stops the in-process runtime from continuing.

Important for `ros` mode:

- the manual edits are applied to the exploration-state occupancy map and frontier logic
- pause and cancel controls stop the exploration loop
- the manual edits are not yet injected into Nav2 as a custom live costmap layer

So in `ros` mode, operator-drawn walls already affect exploration reasoning, but they do not yet become native Nav2 obstacles inside Nav2's internal costmaps.

## Useful ROS/Nav2 Checks

If the ROS stack is not behaving correctly:

```bash
cd /home/alin/Robot42
conda deactivate || true
unset PYTHONHOME
source /opt/ros/humble/setup.bash

ros2 node list
ros2 topic info /map
ros2 topic echo /odom --once
ros2 topic echo /tf --once
ros2 action info /compute_path_to_pose
ros2 action info /navigate_to_pose
```

Expected:

- `/xlerobot_maniskill_ros_bridge` exists
- `/slam_toolbox` exists
- `/compute_path_to_pose` has exactly one action server
- `/navigate_to_pose` has exactly one action server

## Duplicate Nav2 Cleanup

If the goal client or runtime reports multiple Nav2 action servers:

```bash
ps -ef | grep -E 'nav2_bringup|navigation_launch.py|planner_server|controller_server|bt_navigator|behavior_server|lifecycle_manager_navigation' | grep -v grep
pkill -f 'ros2 launch nav2_bringup navigation_launch.py' || true
source /opt/ros/humble/setup.bash
ros2 daemon stop
ros2 daemon start
ros2 node list
ros2 action info /compute_path_to_pose
```

Then restart only the Nav2 terminal.

## Related Files

- [SIMULATION_NAV2.md](/home/alin/Robot42/SIMULATION_NAV2.md)
- [plans/testing_nav2_bridge_connection.md](/home/alin/Robot42/plans/testing_nav2_bridge_connection.md)
