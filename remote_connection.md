# Remote Exploration Connection

This is the setup for running the exploration playground brain on a MacBook while ROS 2 and Nav2 run on an Ubuntu machine over WiFi.

The MacBook does not need to join ROS 2 DDS discovery directly. The intended path is:

```text
MacBook exploration playground -> HTTP -> Ubuntu ROS/Nav2 router -> local ROS 2/Nav2 graph
```

The ROS/Nav2 router is the only process that talks to ROS 2. The playground sends fused map, pose, and scan state to the router over HTTP, and the router republishes those into ROS topics and asks Nav2 for paths.

## Ubuntu ROS/Nav2 Machine

Start the router bound to the WiFi interface, not localhost:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_nav2_adapter_server.py \
  --host 0.0.0.0 \
  --port 8891
```

Keep this terminal running. The important difference from local-only runs is `--host 0.0.0.0`. The default `127.0.0.1` only accepts connections from the same machine.

In another terminal on the same Ubuntu machine, start Nav2:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  use_composition:=False \
  params_file:=/home/alin/Robot42/artifacts/nav2/xlerobot_nav2_params.yaml
```

## MacBook Brain

Find the Ubuntu machine's WiFi IP address, then point the playground at the router:

```bash
python examples/xlerobot_exploration_playground.py \
  --movement-mode ros \
  --ui-flavor developer \
  --explorer-policy llm \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --ros-adapter-url http://<UBUNTU_WIFI_IP>:8891 \
  --ros-navigation-map-source fused_scan \
  --sensor-range-m 10.0 \
  --sim-motion-speed fastest \
  --no-automatic-semantic-waypoints
```

Replace `<UBUNTU_WIFI_IP>` with the actual IP address of the Ubuntu ROS/Nav2 machine.

If Ollama is running on a different machine, change `--llm-base-url` to that machine instead of `http://localhost:11434`.

## Network Checklist

- The MacBook and Ubuntu machine must be on the same trusted LAN or VPN.
- TCP port `8891` must be reachable from the MacBook.
- If a firewall is enabled on Ubuntu, allow inbound TCP traffic on port `8891`.
- The router is unauthenticated HTTP, so do not expose it on an untrusted network.
- If you want to open the playground review UI from another device, run the playground with `--review-host 0.0.0.0`; otherwise the UI defaults to localhost.

Quick connectivity check from the MacBook:

```bash
curl http://<UBUNTU_WIFI_IP>:8891/api/runtime/scan_count
```

If the router is reachable, the request should return JSON rather than a connection error. Some runtime endpoints may still report readiness errors until Nav2 is fully up.

## ROS Checks On Ubuntu

Run these in a ROS-sourced terminal on the Ubuntu machine:

```bash
cd /home/alin/Robot42
source /opt/ros/humble/setup.bash

ros2 topic echo /clock --once
ros2 run tf2_ros tf2_echo map base_link
ros2 topic echo /map --once
ros2 action list | grep compute_path
```

Expected basics:

- `/clock` publishes from the router.
- `map -> base_link` TF resolves.
- `/map` publishes after the playground performs its first scan.
- `compute_path_to_pose` appears after Nav2 is up.

## Notes

- Do not run `examples/xlerobot_nav2_bridge_playground.py` for this router flow.
- The persistent `/map` comes from the playground's fused RGB-D map.
- During motion, the playground sends short-lived scan observations to `/scan` for Nav2's local costmap.
- Nav2 and the router should run on the same ROS 2 environment so the ROS side stays local and predictable.
