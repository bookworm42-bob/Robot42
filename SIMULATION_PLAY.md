# XLeRobot ManiSkill Play Setup

This repository now includes a local bootstrap for running the XLeRobot ManiSkill
keyboard demos without manually copying files into the installed `mani_skill`
package.

## What This Uses

- local XLeRobot sim files from `/Users/alin/xlerobot_forked/simulation/Maniskill`
- an installed `mani_skill` runtime
- runtime registration of:
  - the `xlerobot` robot agent
  - the `SceneManipulation-v1` environment
- the URDF and meshes directly from the forked XLeRobot repo

## One-Time Setup

```bash
cd /Users/alin/multido
./scripts/setup_xlerobot_maniskill_env.sh
```

This creates `.venv-maniskill`, installs the Python dependencies used by the
keyboard demos, and downloads the `ReplicaCAD` asset bundle through ManiSkill.

On macOS, ManiSkill rendering also requires Vulkan to be installed and exported
in your shell environment. The setup script prints a reminder if it detects
macOS.

Important: on the current macOS host, `mani-skill` installs but `sapien` does
not, so the setup cannot complete into a runnable simulator here. The launcher
and bootstrap are ready, but the actual play mode still needs a Linux/Ubuntu
environment with a compatible SAPIEN build.

## Sanity Check

```bash
cd /Users/alin/multido
./scripts/run_xlerobot_maniskill_play.sh --check
```

Expected output includes:

- `registered_agent: xlerobot`
- `registered_env: SceneManipulation-v1`

## Launch The Keyboard Play Mode

Default dual-arm end-effector keyboard control:

```bash
cd /Users/alin/multido
./scripts/run_xlerobot_maniskill_play.sh
```

Equivalent direct command:

```bash
cd /Users/alin/multido
.venv-maniskill/bin/python -m multido_xlerobot.maniskill --demo ee_keyboard
```

## Useful Variants

Joint-control demo:

```bash
./scripts/run_xlerobot_maniskill_play.sh --demo joint_control
```

Camera visualization via Rerun:

```bash
./scripts/run_xlerobot_maniskill_play.sh --demo camera_rerun
```

Explicit environment and render settings:

```bash
./scripts/run_xlerobot_maniskill_play.sh \
  --env-id SceneManipulation-v1 \
  --render-mode human \
  --shader default
```

## Notes

- The safest default environment id is `SceneManipulation-v1`, because it is
  registered locally at runtime from the XLeRobot fork.
- This setup intentionally avoids editing `site-packages/mani_skill`.
- The current launcher focuses on the dual-arm `xlerobot` path. The local fork
  does not currently expose a complete `xlerobot_single` registration in the
  same way, so the play setup does not default to it.
