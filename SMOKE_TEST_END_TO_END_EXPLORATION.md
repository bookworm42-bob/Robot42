# Smoke Test: End-to-End Real Exploration Stack

This runbook starts the real robot brain, offload ROS/Nav2 stack, and the Mode B smoke test.

Mode B means:

```text
robot brain asks offload Nav2 router for current pose
robot brain asks offload Nav2 router for compute_path_to_pose
robot brain follows the path locally with x.vel/theta.vel
robot brain validates final pose through RGB-D odometry
```

Replace these placeholders:

```text
ROBOT_BRAIN_IP = robot brain network address
OFFLOAD_IP = ROS/Nav2 offload computer network address
```

## Robot Brain

### Terminal RB-1: Orbbec RGB-D Sidecar

```bash
cd /home/alin/Robot42

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test -DORBBEC_SDK_ROOT="$HOME/orbbec/sdk"
cmake --build build/orbbec_rgb_test

./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --latest-only \
  --enable-depth \
  --output-dir artifacts/orbbec_rgbd
```

### Terminal RB-2: Robot Brain Agent

Keep the robot wheels raised for the first run.

```bash
cd /home/alin/Robot42

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --orbbec-output-dir artifacts/orbbec_rgbd
```

Check the robot brain locally:

```bash
curl http://127.0.0.1:8765/health
curl -I http://127.0.0.1:8765/rgb
curl -I http://127.0.0.1:8765/depth
```

## Offload ROS/Nav2 Computer

### Terminal OFF-0: Generate SLAM and Nav2 Params

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.real_nav2_config \
  --base-nav2-params /opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml \
  --output-dir /home/alin/Robot42/artifacts/nav2 \
  --scan-topic /scan \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_link \
  --max-laser-range 6.0 \
  --max-linear-velocity 0.03 \
  --max-angular-velocity 0.10 \
  --local-costmap-width 2 \
  --local-costmap-height 2
```

This writes:

```text
/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

### Terminal OFF-1: Real ROS Bridge

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url http://ROBOT_BRAIN_IP:8765 \
  --publish-rate-hz 10 \
  --camera-x-m 0.0 \
  --camera-y-m 0.0 \
  --camera-z-m 0.35 \
  --camera-yaw-rad 0.0
```

### Terminal OFF-2: RGB-D Visual Odometry

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --odom-topic /odom \
  --publish-rate-hz 15
```

### Terminal OFF-3: SLAM Toolbox

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=false \
  slam_params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_slam_toolbox.yaml
```

### Terminal OFF-4: Nav2

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

### Terminal OFF-5: Nav2 HTTP Router

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

python -m xlerobot_playground.ros_nav2_router \
  --host 0.0.0.0 \
  --port 8891 \
  --no-publish-clock \
  --no-publish-external-state-tf
```

## Verification

Run these on the offload computer:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /scan --once
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo map odom
ros2 action list | grep compute_path_to_pose
```

Run these from the robot brain:

```bash
curl http://OFFLOAD_IP:8891/api/health
curl http://OFFLOAD_IP:8891/api/router/current_pose
```

Run this no-motion preflight from the robot brain before the real smoke test:

```bash
python -m xlerobot_playground.robot_brain_smoke_test \
  --router-url http://OFFLOAD_IP:8891 \
  --robot-brain-url http://127.0.0.1:8765 \
  --preflight-only
```

This checks the offload router, local robot brain agent, and RGB-D odometry pose path without sending wheel commands.

## Run The Smoke Test From Robot Brain

### Terminal RB-3: Mode B Smoke Test

```bash
cd /home/alin/Robot42

python -m xlerobot_playground.robot_brain_smoke_test \
  --router-url http://OFFLOAD_IP:8891 \
  --robot-brain-url http://127.0.0.1:8765 \
  --forward-m 0.05 \
  --turn-deg 5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.10 \
  --goal-distance-tolerance-m 0.025 \
  --yaw-tolerance-deg 2.5 \
  --max-translation-error-m 0.05 \
  --max-yaw-error-deg 5 \
  --step-timeout-s 15
```

Expected success:

```json
{
  "ok": true,
  "mode": "robot_brain_follows_nav2_path"
}
```

The smoke test executes:

```text
forward 5cm
rotate left 5deg
rotate right 5deg
```

## What To Watch

- `/depth` should be non-empty on the robot brain.
- `/scan` should not be all zero or all max range.
- `/odom` should be stable when stationary.
- `/odom` should move in the expected direction when the robot moves.
- The robot brain agent should stop after every smoke-test step.
- Nav2 should expose `compute_path_to_pose`.
- The final smoke-test JSON should report small translation and yaw errors.

## Current Missing Pieces

- Real validation of Orbbec depth and RGB-D odometry quality.
- Real measurement for `base_link -> head_camera_link`.
- No-Nav2 calibration script is still only a TODO.
- Nav2 parameters may need tuning after the first real run.
