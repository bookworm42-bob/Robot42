# Run Nav2 Router Playground

This is the current intended architecture:

- The exploration playground owns ManiSkill, RGB-D scan fusion, map building, frontier selection, and the UI.
- The ROS/Nav2 adapter is only a router: it receives map/pose/scan state from the playground, publishes ROS topics, and asks Nav2 for paths.
- Nav2 consumes the router-published map/TF/scan and returns planned paths.
- The persistent `/map` comes from the playground's fused RGB-D map. During movement, the playground sends short-lived scan observations to `/scan` for Nav2's local costmap without merging those observations back into the persistent map.
- Do not run `examples/xlerobot_nav2_bridge_playground.py` for this flow.

## Terminal 1: Start The ROS/Nav2 Router

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_nav2_adapter_server.py \
  --host 127.0.0.1 \
  --port 8891
```

Keep this terminal running.

## Terminal 2: Start Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

Keep this terminal running.

## Terminal 3: Start The Exploration Playground

```bash
cd /home/alin/Robot42
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_exploration_playground.py \
  --movement-mode ros \
  --ui-flavor developer \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --ros-adapter-url http://127.0.0.1:8891 \
  --ros-navigation-map-source fused_scan \
  --sensor-range-m 10.0 \
  --sim-motion-speed fastest \
  --ros-manual-spin-angular-speed-rad-s 0.25 \
  --no-automatic-semantic-waypoints \
  --use-keyboard-controls \
  --keyboard-speed fast
```

The UI should open at:

```text
http://127.0.0.1:8770/
```

The playground terminal should print:

```text
Pumping the ManiSkill/SAPIEN viewer on the main thread to keep the window responsive.
```

## Optional: Start Ollama First

Use this only if Ollama is not already running:

```bash
ollama serve
```

Make sure the model exists:

```bash
ollama list
```

## Quick ROS Checks

Run these in a separate ROS-sourced terminal if Nav2 does not behave as expected:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 topic echo /clock --once
ros2 run tf2_ros tf2_echo map base_link
ros2 topic echo /map --once
ros2 action list | grep compute_path
```

Expected basics:

- `/clock` publishes from the adapter/router.
- `map -> base_link` TF resolves.
- `/map` publishes after the playground performs its first scan.
- `compute_path_to_pose` appears in the action list after Nav2 is up.

## Current Behavior Notes

- The robot scan rotation is slowed with `--ros-manual-spin-angular-speed-rad-s 0.25`.
- `--sim-motion-speed` controls the whole simulated motion speed profile:
  - `normal`: realistic-ish debug speed, about `0.30 m/s` path following and base turn speed.
  - `faster`: about `0.60 m/s` path following and `1.8x` base turn speed.
  - `fastest`: about `1.00 m/s` path following and `3.0x` base turn speed.
- The base turn speed for `normal` is still set by `--ros-manual-spin-angular-speed-rad-s`.
- Nav2 costmap inflation is disabled in `artifacts/nav2/xlerobot_nav2_params.yaml`.
- Nav2 uses a centered rectangular `footprint` polygon as the collision model: `0.3913 m` long by `0.459 m` wide.
- Nav2's global costmap uses the persistent static map only. Nav2's local costmap consumes `/scan` with short observation persistence, so local obstacle observations are refreshed and discarded during navigation.
- Movement in ManiSkill follows Nav2 path poses kinematically. It should no longer jump directly to the frontier.
- While following a path, the playground checks each RGB-D-derived local scan against the swept rectangular base footprint aligned to the current path segment. Known static map obstacles such as door frames are ignored by this emergency guard; only newly observed blockers stop ManiSkill, add a temporary single-cell obstacle overlay for replanning, and ask Nav2 for a fresh path to the same target.
- If Nav2 returns a path that crosses a known occupied map cell, the playground rejects that candidate target and tries another valid target pose near the same frontier before failing the move.
