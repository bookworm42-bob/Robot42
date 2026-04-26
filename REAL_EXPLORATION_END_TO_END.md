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

`real_agentic_exploration` now starts from a clean in-memory map by default even when `--persist-path` points to an existing JSON. Pass `--restore-persisted-state` only when you explicitly want to resume the backend/UI snapshot from that file.

## Robot Brain

Run these on the robot brain Mac. Keep both terminals running.

### Terminal RB-1: Robot Brain Agent

Use the Python environment that has LeRobot/XLeRobot installed.

```bash
cd /Users/alin/Robot42
conda activate xlerobot
python -m pip install aiohttp

python -m xlerobot_playground.robot_brain_agent \
  --allow-motion-commands \
  --debug-motion \
  --robot-kind xlerobot_2wheels \
  --port1 /dev/tty.usbmodem5B140330101 \
  --port2 /dev/tty.usbmodem5B140332271 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.30
```

### Terminal RB-2: Orbbec Sidecar

```bash
cd /Users/alin/Robot42

cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test
cmake --build build/orbbec_rgb_test

sudo ./build/orbbec_rgb_test/orbbec_rgb_test --list-profiles

sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-depth-registration \
  --enable-imu \
  --imu-udp-host 127.0.0.1 \
  --imu-udp-port 8766 \
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd
```

If the Orbbec default aligned depth profile is too heavy, use the listed Gemini 2 `Y16 640x400@30` depth source profile:

```bash
sudo ./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-depth-registration \
  --enable-imu \
  --depth-width 640 \
  --depth-height 400 \
  --depth-fps 30 \
  --imu-udp-host 127.0.0.1 \
  --imu-udp-port 8766 \
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd
```

Quick health checks from the robot brain:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/rgb --output /tmp/brain_rgb.ppm
curl http://127.0.0.1:8765/depth --output /tmp/brain_depth.pgm
curl http://127.0.0.1:8765/imu
python - <<'PY'
import asyncio
from aiohttp import ClientSession

async def main():
    async with ClientSession() as session:
        async with session.ws_connect("ws://127.0.0.1:8765/ws/imu") as ws:
            first = await ws.receive()
            print(first.data)

asyncio.run(main())
PY
```

`/imu` is now an in-memory debug snapshot. The high-rate IMU path is `Orbbec callback -> UDP datagram -> robot_brain_agent memory -> /ws/imu websocket`. `latest_imu.json` is no longer used in the high-rate path.

## Offload Computer

Run these on the ROS/Nav2 offload computer. Keep terminals OC-1, OC-2, OC-4, and OC-5 running. OC-3 only needs to be run when generating/updating the Nav2 params.

### Terminal OC-1: Real ROS Bridge

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate
python -m pip install aiohttp

python -m xlerobot_playground.real_ros_bridge \
  --robot-brain-url http://192.168.1.133:8765 \
  --publish-rate-hz 30 \
  --cmd-vel-timeout-s 0.5 \
  --max-linear-m-s 0.03 \
  --max-angular-rad-s 0.30 \
  --camera-x-m 0.0 \
  --camera-y-m 0.0 \
  --camera-z-m 0.35 \
  --camera-yaw-rad 0.0 \
  --allow-motion-commands
```

This publishes camera images, depth-derived `/scan`, `/imu`, camera transforms, and forwards ROS `/cmd_vel` to the robot brain.

`/imu` is a raw `sensor_msgs/Imu` stream carrying both angular velocity and linear acceleration. In robot-brain mode it is now pushed over a persistent websocket, so `/imu` is no longer capped by the old poll timer.

Quick checks:

```bash
ros2 topic echo /camera/head/camera_info --once
ros2 topic echo /scan --once
ros2 topic echo /imu --once
ros2 topic hz /imu
curl http://ROBOT_BRAIN_IP:8765/health
```

Migration note:

- Remove any process that depends on `latest_imu.json`. The sidecar no longer writes it.
- Keep `/cmd_vel` and `/stop` on HTTP. IMU now comes from `ws://ROBOT_BRAIN_IP:8765/ws/imu`.
- If the brain and sidecar run on different hosts, change both `--imu-udp-host` and `--imu-udp-port` together.

Verification plan:

- On the robot brain, watch `orbbec_rgb_test` for `IMU callback rate ~= ... Hz`.
- On the robot brain, watch `robot_brain_agent` for `IMU rx rate~=...Hz` and `IMU ws send rate~=...Hz`.
- On the offload computer, watch `real_ros_bridge` for `IMU websocket connected` and `IMU stream rx~=...Hz publish~=...Hz`.
- Run `ros2 topic hz /imu` and confirm the rate is no longer limited to the old tens-of-Hz poll ceiling.
- If the callback rate is still low, audit the current Orbbec aggregate mode `OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE` first. The transport bottleneck is removed, but that SDK aggregation setting can still quantize the upstream callback cadence.

### Terminal OC-2: IMU Yaw Filter

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.imu_yaw_filter \
  --imu-topic /imu \
  --output-topic /imu/filtered_yaw \
  --input-frame-convention camera_optical \
  --yaw-source gyro_y \
  --bias-calibration-s 0.5 \
  --yaw-rate-lowpass-alpha 0.2
```

This matches the OrbbecViewer CSV path: integrate corrected `gyro_y`.
Keep the robot still for the first 0.5 seconds after startup so the yaw filter calibrates gyro bias cleanly.

Quick checks:

```bash
ros2 topic echo /imu/filtered_yaw --once
```

### Terminal OC-3: RGB-D Visual Odometry

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python -m xlerobot_playground.rgbd_visual_odometry \
  --rgb-topic /camera/head/image_raw \
  --depth-topic /camera/head/depth/image_raw \
  --camera-info-topic /camera/head/camera_info \
  --imu-topic /imu/filtered_yaw \
  --odom-topic /odom \
  --imu-frame-convention base_link \
  --imu-bias-calibration-s 0.0 \
  --publish-rate-hz 30
```

This consumes the filtered yaw IMU topic, which is already bias-corrected and expressed in `base_link`.

Quick checks:

```bash
ros2 topic echo /odom --once
ros2 run tf2_ros tf2_echo odom base_link
```

### Terminal OC-4: Nav2 Params

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

### Terminal OC-5: Nav2

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

### Terminal OC-6: Real Exploration UI And Loop

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
  --ros-manual-spin-angular-speed-rad-s 0.30 \
  --max-decisions 8 \
  --ros-imu-topic /imu/filtered_yaw \
  --stop-after-initial-scan
```

Open the UI from your browser:

```text
http://OFFLOAD_IP:8770
```

Click `Start Explore` in the UI. The robot should begin with a faster right-turn 360 degree scan, build a partial occupancy map, detect frontiers, preview Nav2 paths, choose a frontier, and send a Nav2 navigation goal.

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
  --ros-manual-spin-angular-speed-rad-s 0.30 \
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

During the initial scan, the robot should rotate to the right in place and the UI should move from an empty/not-started map to a partial occupancy map with candidate frontiers.

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
