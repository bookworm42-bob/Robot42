# Real XLeRobot Agentic Exploration Commands

This is the standalone command runbook for real exploration. It runs the same frontier/LLM exploration loop as the ManiSkill/ROS path, but with the real robot providing RGB-D, `/scan`, `/odom`, and motor execution through the robot brain bridge.

This full exploration path does **not** use the HTTP Nav2 router from the smoke tests. In this mode:

```text
exploration loop -> Nav2 navigate_to_pose -> /cmd_vel -> real_ros_bridge -> robot brain -> XLeRobot motors
```

So Nav2 does both path planning and local velocity control. The robot brain executes safe body velocity commands.

Replace these placeholders:

```text
ROBOT_BRAIN_IP   IP address of the Mac/robot brain running robot_brain_agent
OFFLOAD_IP       IP address of the ROS/Nav2 offload computer
YOUR_MODEL       LLM model name, only for LLM mode
YOUR_API_KEY     LLM API key, only for LLM mode
```

## What This Runs

```text
robot brain Orbbec sidecar
robot brain HTTP agent
offload real_ros_bridge
offload RGB-D visual odometry
offload Nav2
offload real_agentic_exploration + web UI
```

The exploration session does:

```text
initial 360 degree scan
scan fusion into an occupancy map when using fused_scan
frontier detection
Nav2 path preview for frontier candidates
LLM or heuristic frontier choice
Nav2 navigate_to_pose for the selected frontier
repeat scan/frontier/decision/navigation
save the final map JSON
```

## Robot Brain

Run these on the robot brain Mac. Keep both terminals running.

### Terminal RB-1: Orbbec Sidecar

```bash
cd /Users/alin/Robot42

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test
cmake --build build/orbbec_rgb_test

sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --latest-only \
  --enable-depth \
  --output-dir artifacts/orbbec_rgbd
```

If the Orbbec rejects the default depth profile, use the explicit profile that worked for Gemini 2:

```bash
sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --latest-only \
  --enable-depth \
  --depth-width 640 \
  --depth-height 576 \
  --depth-fps 30 \
  --output-dir artifacts/orbbec_rgbd
```

### Terminal RB-2: Robot Brain Agent

Use the Python environment that has LeRobot/XLeRobot installed.

```bash
cd /Users/alin/Robot42
conda activate xlerobot

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --debug-motion \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --orbbec-output-dir artifacts/orbbec_rgbd
```

Quick health checks from the robot brain:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/rgb --output /tmp/brain_rgb.ppm
curl http://127.0.0.1:8765/depth --output /tmp/brain_depth.pgm
```

## Offload Computer

Run these on the ROS/Nav2 offload computer. Keep terminals OC-1, OC-2, OC-4, and OC-5 running. OC-3 only needs to be run when generating/updating the Nav2 params.

### Terminal OC-1: Real ROS Bridge

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url http://192.168.1.133:8765 \
  --publish-rate-hz 10 \
  --cmd-vel-timeout-s 0.5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --camera-x-m 0.0 \
  --camera-y-m 0.0 \
  --camera-z-m 0.35 \
  --camera-yaw-rad 0.0
```

This publishes camera images, depth-derived `/scan`, camera transforms, and forwards ROS `/cmd_vel` to the robot brain.

Quick checks:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /scan --once
curl http://ROBOT_BRAIN_IP:8765/health
```

### Terminal OC-2: RGB-D Visual Odometry

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --odom-topic /odom \
  --publish-rate-hz 15
```

Quick checks:

```bash
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

### Terminal OC-3: Nav2 Params

Generate the conservative Nav2 params once:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_nav2_config \
  --base-nav2-params /opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml \
  --output-dir /home/alin/Robot42/artifacts/nav2 \
  --scan-topic /scan \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_link \
  --max-laser-range 4.0 \
  --max-linear-velocity 0.03 \
  --max-angular-velocity 0.10 \
  --local-costmap-width 2 \
  --local-costmap-height 2
```

### Terminal OC-4: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

Quick checks:

```bash
ros2 action list
ros2 action list | grep compute_path_to_pose
ros2 action list | grep navigate_to_pose
```

### Terminal OC-5: Real Exploration UI And Loop

Start with heuristic policy first. This tests mapping, frontiers, path previews, Nav2 goal execution, and UI without spending LLM calls.

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.real_agentic_exploration \
  --persist-path /home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json \
  --session real_house_v1 \
  --explorer-policy heuristic \
  --serve-review-ui \
  --review-host 0.0.0.0 \
  --review-port 8770 \
  --ros-navigation-map-source fused_scan \
  --ros-ready-timeout-s 30 \
  --ros-turn-scan-timeout-s 75 \
  --ros-manual-spin-angular-speed-rad-s 0.10 \
  --max-decisions 8
```

Open the UI from your browser:

```text
http://OFFLOAD_IP:8770
```

Click `Start Explore` in the UI. The robot should begin with a slow 360 degree scan, build a partial occupancy map, detect frontiers, preview Nav2 paths, choose a frontier, and send a Nav2 navigation goal.

By default this real-exploration command waits for the UI start request before moving the robot. Use `--no-wait-for-ui-start` only when you want the 360 degree scan to begin immediately after the terminal command starts.

Once heuristic exploration is sane, switch to LLM policy:

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --persist-path /home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json \
  --session real_house_v1 \
  --explorer-policy llm \
  --llm-provider openai \
  --llm-model YOUR_MODEL \
  --llm-api-key YOUR_API_KEY \
  --serve-review-ui \
  --review-host 0.0.0.0 \
  --review-port 8770 \
  --ros-navigation-map-source fused_scan \
  --ros-ready-timeout-s 30 \
  --ros-turn-scan-timeout-s 75 \
  --ros-manual-spin-angular-speed-rad-s 0.10 \
  --max-decisions 8
```

## Preflight Checklist

Run these on the offload computer before starting exploration:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /camera/head/image_raw --once
ros2 topic echo /camera/head/depth/image_raw --once
ros2 topic echo /scan --once
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo map odom
ros2 action list
```

Expected action servers include:

```text
/compute_path_to_pose
/navigate_to_pose
```

During the initial scan, the robot should rotate slowly in place and the UI should move from an empty/not-started map to a partial occupancy map with candidate frontiers.

If the map starts but the robot does not rotate, check whether the scan command is being published and forwarded:

```bash
ros2 topic echo /cmd_vel --once
curl http://ROBOT_BRAIN_IP:8765/health
```

The `real_ros_bridge` terminal should log motion forwarding errors if the robot brain rejects `/cmd_vel`.

## What You Should See

- Robot brain logs showing `/rgb`, `/depth`, and `/cmd_vel` requests.
- `real_ros_bridge` publishing `/camera/head/*`, `/scan`, and forwarding `/cmd_vel`.
- RGB-D visual odometry publishing `/odom` and `odom -> base_link`.
- Nav2 accepting `compute_path_to_pose` and `navigate_to_pose`.
- UI showing:
  - current robot pose
  - partial occupancy map
  - frontier candidates
  - recent RGB keyframes
  - selected frontier and Nav2 path/result
- Final map saved to:

```text
/home/alin/Robot42/artifacts/real_xlerobot_exploration_map.json
```

## Current Limitations

- RGB-D visual odometry is experimental and may drift, especially during rotation.
- The initial 360 scan uses ROS `/cmd_vel` through `real_ros_bridge`; the robot brain executes the velocity commands.
- Frontier navigation uses Nav2 `navigate_to_pose`; this is the existing ROS exploration execution path.
- Exact region naming and semantic waypoint quality depend on good RGB keyframes and LLM/VLM configuration.
- Saving the final map JSON works through `--persist-path`; saving directly into persistent robot memory is still a separate integration step.
