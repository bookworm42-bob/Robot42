# XLeRobot Integration

This repo integrates with the fork at `/Users/alin/xlerobot_forked` through the local package `multido_xlerobot`.
Keep that fork synced with the upstream/current XLeRobot checkout, then let Robot42 consume `xlerobot_forked`.

The adapter does not copy XLeRobot code. Instead it:

- validates the fork layout
- requires a Python environment where `lerobot` is installed
- registers XLeRobot extension modules into the `lerobot` namespace
- exposes a small stable facade for robot, VR, model, and recorder access

## Quick Start

```python
from multido_xlerobot import XLeRobotInterface

api = XLeRobotInterface("/Users/alin/xlerobot_forked")
print(api.summary())

robot = api.make_robot()
vr_teleop = api.make_vr_teleop()
record = api.record_module()
```

## Available Facade Methods

- `bootstrap()`
- `modules()`
- `robot_classes()`
- `robot_2wheels_classes()`
- `vr_classes()`
- `model_classes()`
- `record_module()`
- `make_robot_config()`
- `make_robot()`
- `make_2wheels_robot_config()`
- `make_2wheels_robot()`
- `make_vr_config()`
- `make_vr_teleop()`
- `summary()`

## Important Constraint

The XLeRobot fork is currently structured as an extension to `lerobot`, not as a standalone package. The adapter therefore expects:

1. the XLeRobot fork to exist at the configured path
2. a Python environment with `lerobot` already installed

If `lerobot` is missing, bootstrap will fail with a clear error message.

Set `XLEROBOT_FORKED_ROOT` only if the local fork moves. Otherwise the default resolver prefers `~/xlerobot_forked`.
