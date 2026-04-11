# Testing Interactive LLM Frontier Playground

This playground is for testing whether the LLM can choose useful exploration regions from partial RGB-D frontier information.

It intentionally does not use Nav2. It has two backends:

- `synthetic`: fast mocked apartment map and mocked RGB-D thumbnails.
- `maniskill`: actual ManiSkill scene and actual XLeRobot head RGB-D frames, with teleport-only movement after operator review.

The web UI flow is:

- inspect the current scanned 2D occupancy map
- inspect the exact prompt sent to the LLM
- click `Call LLM`
- inspect the structured response and selected frontier on the 2D map
- click `Move To Selected Frontier`
- the playground applies a direct mock pose update or ManiSkill teleport and performs another 360 degree scan

## Mock LLM

```bash
cd /home/alin/Robot42

python examples/xlerobot_interactive_exploration_playground.py \
  --backend synthetic \
  --llm-provider mock \
  --llm-model mock \
  --host 127.0.0.1 \
  --port 8781
```

Open:

```text
http://127.0.0.1:8781/
```

## Local Ollama

Make sure Ollama is running and the model is available:

```bash
ollama list
```

Then run:

```bash
cd /home/alin/Robot42

python examples/xlerobot_interactive_exploration_playground.py \
  --backend synthetic \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --host 127.0.0.1 \
  --port 8781
```

The Ollama provider calls `/api/generate` directly. Recent RGB-D keyframe thumbnails are included as images when present, and the prompt tells the model that frontier information is partial RGB-D-derived map evidence, not complete apartment knowledge.

## ManiSkill RGB-D Teleport Backend

Use this when you want the LLM prompt and images to come from the actual XLeRobot ManiSkill scene, but still want no Nav2 and manual decision gating:

```bash
cd /home/alin/Robot42

source /opt/ros/humble/setup.bash
source /home/alin/Robot42/.venv-maniskill/bin/activate

python examples/xlerobot_interactive_exploration_playground.py \
  --backend maniskill \
  --repo-root /home/alin/XLeRobot \
  --env-id SceneManipulation-v1 \
  --robot-uid xlerobot \
  --render-mode human \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --llm-base-url http://localhost:11434 \
  --host 127.0.0.1 \
  --port 8781
```

By default, the initial scan is a 360 turnaround implemented as yaw teleports at the current base position. The map is a deterministic 2D projection from the XLeRobot head depth image, and recent RGB frames are attached to the LLM request. Clicking `Move To Selected Frontier` teleports the robot to the selected frontier's validated inward approach pose and scans again.

For startup scan experiments, you can bypass the stitched 360 scan and rotate the default spawn in place:

```bash
python examples/xlerobot_interactive_exploration_playground.py \
  --backend maniskill \
  --repo-root /home/alin/XLeRobot \
  --env-id SceneManipulation-v1 \
  --robot-uid xlerobot \
  --render-mode human \
  --spawn-facing front \
  --scan-mode front_only
```

Use `--spawn-facing left`, `--spawn-facing right`, or `--spawn-facing back` to rotate the default ManiSkill spawn by `-90`, `+90`, or `180` degrees before scanning. `--scan-mode front_only` keeps each scan to the current forward view instead of doing the 360 turnaround stitching.

For the ManiSkill backend, the playground runs the web server in a background thread and pumps the SAPIEN viewer on the main thread. If the SAPIEN window still looks paused while Ollama is generating or while a scan is running, wait for the current operation to finish; those operations intentionally lock simulator state while they read RGB-D or update the robot pose.

Frontier placement is split into two concepts:

- `frontier_boundary_pose`: the boundary evidence where known-free projected RGB-D cells touch unknown cells.
- `nav_pose` / `approach_pose`: the robot placement pose, offset back onto the known-free side of the boundary and validated against a conservative XLeRobot/IKEA cart footprint.

The footprint radius uses the simulator URDF base collision box dimensions, about `0.3913 m x 0.459 m`, plus padding. This makes the teleport target conservative instead of placing the robot exactly on the frontier edge.

Useful tuning flags:

```bash
--scan-yaw-samples 12
--depth-beam-stride 2
--max-frontiers 12
--teleport-settle-steps 1
--build-config-idx 0
--spawn-facing front
--scan-mode turnaround
--spawn-x 0.0 --spawn-y 0.0 --spawn-yaw 0.0
```

## What This Does Not Test

This does not test Nav2, obstacle avoidance, or controller tuning. In the ManiSkill backend, it also does not prove that a selected point is reachable by a controller, because movement is teleport-only by design.

Use the Nav2 bridge runbook for that:

```text
plans/testing_nav2_bridge_connection.md
```
