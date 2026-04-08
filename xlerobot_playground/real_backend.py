from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from multido_xlerobot import XLeRobotInterface
from multido_xlerobot.bootstrap import bootstrap_xlerobot


@dataclass(frozen=True)
class CameraSpec:
    name: str
    driver: str
    source: str


@dataclass
class RecordingSession:
    dataset: Any
    task: str
    active: bool = False


@dataclass(frozen=True)
class VRRecordingControls:
    toggle_recording: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


@dataclass(frozen=True)
class VRRecordingDecision:
    start_recording: bool = False
    save_episode: bool = False
    discard_episode: bool = False
    quit_session: bool = False
    reset_robot: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend launcher for XLeRobot real teleop and local LeRobot recording."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manipulate = subparsers.add_parser("manipulate", help="Launch real teleoperation.")
    _add_shared_args(manipulate)
    manipulate.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")

    record = subparsers.add_parser("record", help="Launch real teleop with local LeRobot recording.")
    _add_shared_args(record)
    record.add_argument("--controller", choices=("keyboard", "vr"), default="keyboard")
    record.add_argument("--dataset-id", default="local/xlerobot_playground")
    record.add_argument("--dataset-root", default="./datasets")
    record.add_argument("--task", default="XLeRobot teleoperation")
    record.add_argument("--use-videos", action="store_true")
    record.add_argument("--start-key", default="[")
    record.add_argument("--stop-key", default="]")
    record.add_argument("--quit-key", default="\\")
    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(Path.home() / "XLeRobot"))
    parser.add_argument("--port1", default="/dev/ttyACM0")
    parser.add_argument("--port2", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=DRIVER:SOURCE",
        help=(
            "Camera config. Example: `head=realsense:125322060037` or "
            "`left_wrist=opencv:/dev/video0`."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument("--xlevr-path", default=None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_xlerobot(args.repo_root)

    interface = XLeRobotInterface(args.repo_root)
    config_cls, robot_cls = interface.robot_classes()
    robot_config = config_cls(
        port1=args.port1,
        port2=args.port2,
        cameras=_build_camera_configs(
            args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        use_degrees=args.use_degrees,
    )
    robot = robot_cls(robot_config)

    recording = None
    if args.mode == "record":
        recording = RecordingSession(
            dataset=_create_dataset(
                robot,
                dataset_id=args.dataset_id,
                dataset_root=args.dataset_root,
                fps=args.fps,
                use_videos=args.use_videos,
            ),
            task=args.task,
        )

    if args.controller == "keyboard":
        return _run_keyboard_backend(
            repo_root=Path(args.repo_root).expanduser().resolve(),
            robot=robot,
            fps=args.fps,
            recording=recording,
            start_key=getattr(args, "start_key", "["),
            stop_key=getattr(args, "stop_key", "]"),
            quit_key=getattr(args, "quit_key", "\\"),
        )
    return _run_vr_backend(
        interface=interface,
        robot=robot,
        fps=args.fps,
        recording=recording,
        start_key=getattr(args, "start_key", "["),
        stop_key=getattr(args, "stop_key", "]"),
        quit_key=getattr(args, "quit_key", "\\"),
        xlevr_path=args.xlevr_path,
    )


def _build_camera_configs(
    camera_specs: list[str],
    *,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras: dict[str, Any] = {}
    for raw_spec in camera_specs:
        spec = _parse_camera_spec(raw_spec)
        if spec.driver == "opencv":
            source: Any = int(spec.source) if spec.source.isdigit() else spec.source
            cameras[spec.name] = OpenCVCameraConfig(
                index_or_path=source,
                fps=fps,
                width=width,
                height=height,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            continue
        if spec.driver == "realsense":
            cameras[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=spec.source,
                fps=fps,
                width=width,
                height=height,
                color_mode=ColorMode.BGR,
                rotation=Cv2Rotation.NO_ROTATION,
                use_depth=True,
            )
            continue
        raise ValueError(f"Unsupported camera driver `{spec.driver}` in `{raw_spec}`")
    return cameras


def _parse_camera_spec(raw_spec: str) -> CameraSpec:
    if "=" not in raw_spec or ":" not in raw_spec:
        raise ValueError(
            f"Invalid camera spec `{raw_spec}`. Use `NAME=DRIVER:SOURCE`, "
            "for example `head=realsense:125322060037`."
        )
    name, remainder = raw_spec.split("=", 1)
    driver, source = remainder.split(":", 1)
    return CameraSpec(name=name.strip(), driver=driver.strip(), source=source.strip())


def _run_keyboard_backend(
    *,
    repo_root: Path,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    import numpy as np

    keyboard_module = _load_example_module(
        repo_root / "software" / "examples" / "4_xlerobot_teleop_keyboard.py",
        "xlerobot_playground._real_keyboard_example",
    )

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    previous_pressed_keys: set[str] = set()

    robot.connect()
    init_rerun(session_name="xlerobot_real_keyboard_playground")
    keyboard.connect()

    obs = robot.get_observation()
    kin_left = keyboard_module.SO101Kinematics()
    kin_right = keyboard_module.SO101Kinematics()
    left_arm = keyboard_module.SimpleTeleopArm(kin_left, keyboard_module.LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = keyboard_module.SimpleTeleopArm(kin_right, keyboard_module.RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = keyboard_module.SimpleHeadControl(obs)

    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)
    _print_recording_guide(recording, start_key=start_key, stop_key=stop_key, quit_key=quit_key)

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(keyboard.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            left_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.LEFT_KEYMAP.items()
            }
            right_key_state = {
                action: (key in pressed_keys) for action, key in keyboard_module.RIGHT_KEYMAP.items()
            }

            if left_key_state.get("triangle"):
                left_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if right_key_state.get("triangle"):
                right_arm.execute_rectangular_trajectory(robot, fps=fps)
                continue
            if left_key_state.get("reset"):
                left_arm.move_to_zero_position(robot)
                continue
            if right_key_state.get("reset"):
                right_arm.move_to_zero_position(robot)
                continue
            if "?" in pressed_keys:
                head_control.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot)
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **head_action, **base_action}
            sent_action = robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        try:
            robot.disconnect()
        finally:
            if keyboard.is_connected:
                keyboard.disconnect()
    return 0


def _run_vr_backend(
    *,
    interface: XLeRobotInterface,
    robot: Any,
    fps: int,
    recording: RecordingSession | None,
    start_key: str,
    stop_key: str,
    quit_key: str,
    xlevr_path: str | None,
) -> int:
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    hotkeys = KeyboardTeleop(KeyboardTeleopConfig())
    previous_pressed_keys: set[str] = set()

    robot.connect()
    init_rerun(session_name="xlerobot_real_vr_playground")
    hotkeys.connect()

    vr_overrides = {}
    if xlevr_path is not None:
        vr_overrides["xlevr_path"] = xlevr_path
    vr_teleop = interface.make_vr_teleop(**vr_overrides)
    vr_teleop.connect(robot=robot)
    vr_teleop.send_feedback()
    robot.send_action(vr_teleop.move_to_zero_position(robot))
    _print_recording_guide(
        recording,
        start_key=start_key,
        stop_key=stop_key,
        quit_key=quit_key,
        controller="vr",
    )

    try:
        while True:
            start_loop_t = time.perf_counter()
            try:
                pressed_keys = set(hotkeys.get_action().keys())
            except DeviceNotConnectedError:
                break
            newly_pressed = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys
            if quit_key in newly_pressed:
                break
            _handle_recording_hotkeys(
                recording,
                newly_pressed,
                start_key=start_key,
                stop_key=stop_key,
            )

            vr_decision = VRRecordingDecision()
            if recording is not None:
                vr_decision = _decide_vr_recording_action(
                    recording.active,
                    _map_vr_events_to_recording_controls(vr_teleop.get_vr_events()),
                )
                _apply_vr_recording_decision(recording, vr_decision)
                if vr_decision.quit_session:
                    break

            obs = robot.get_observation()
            if vr_decision.reset_robot:
                action = vr_teleop.move_to_zero_position(robot)
            else:
                action = vr_teleop.get_action(obs, robot)
            if action:
                sent_action = robot.send_action(action)
            else:
                sent_action = {}
            obs = robot.get_observation()
            log_rerun_data(obs, sent_action)
            _record_frame_if_needed(recording, obs, sent_action)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / fps - dt_s))
    finally:
        _finalize_recording(recording)
        try:
            robot.disconnect()
        finally:
            try:
                vr_teleop.disconnect()
            except Exception:
                pass
            if hotkeys.is_connected:
                hotkeys.disconnect()
    return 0


def _create_dataset(
    robot: Any,
    *,
    dataset_id: str,
    dataset_root: str,
    fps: int,
    use_videos: bool,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}
    return LeRobotDataset.create(
        dataset_id,
        fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=use_videos,
    )


def _record_frame_if_needed(recording: RecordingSession | None, observation: dict[str, Any], action: dict[str, Any]) -> None:
    if recording is None or not recording.active:
        return

    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    observation_frame = build_dataset_frame(recording.dataset.features, observation, prefix=OBS_STR)
    complete_action = dict(action)
    for action_name in recording.dataset.features[ACTION]["names"]:
        if action_name in complete_action:
            continue
        if action_name in observation:
            complete_action[action_name] = observation[action_name]
        elif action_name.endswith(".vel"):
            complete_action[action_name] = 0.0
        else:
            complete_action[action_name] = 0.0

    action_frame = build_dataset_frame(recording.dataset.features, complete_action, prefix=ACTION)
    frame = {
        **observation_frame,
        **action_frame,
        "task": recording.task,
        "timestamp": time.time(),
    }
    recording.dataset.add_frame(frame)


def _handle_recording_hotkeys(
    recording: RecordingSession | None,
    pressed_keys: set[str],
    *,
    start_key: str,
    stop_key: str,
) -> None:
    if recording is None:
        return

    if start_key in pressed_keys and not recording.active:
        recording.active = True
        print(f"Recording started. Press `{stop_key}` to save the current episode.")
        return

    if stop_key in pressed_keys and recording.active:
        _save_episode(recording)


def _save_episode(recording: RecordingSession) -> None:
    if not _episode_buffer_has_frames(recording.dataset):
        recording.active = False
        print("Recording stopped. No frames captured, skipping save.")
        return

    recording.dataset.save_episode()
    recording.active = False
    print(f"Saved episode {recording.dataset.meta.total_episodes - 1}.")


def _finalize_recording(recording: RecordingSession | None) -> None:
    if recording is None or not recording.active:
        return
    print("Saving the active episode before exit.")
    _save_episode(recording)


def _discard_episode(recording: RecordingSession) -> None:
    clear_episode_buffer = getattr(recording.dataset, "clear_episode_buffer", None)
    if callable(clear_episode_buffer):
        clear_episode_buffer()
    else:
        buffer = getattr(recording.dataset, "episode_buffer", None)
        if isinstance(buffer, dict):
            for key, value in buffer.items():
                if key == "size":
                    buffer[key] = 0
                elif hasattr(value, "clear"):
                    value.clear()
    recording.active = False
    print("Discarded the current episode.")


def _episode_buffer_has_frames(dataset: Any) -> bool:
    buffer = getattr(dataset, "episode_buffer", None)
    return bool(buffer and buffer.get("size", 0) > 0)


def _print_recording_guide(
    recording: RecordingSession | None,
    *,
    start_key: str,
    stop_key: str,
    quit_key: str,
    controller: str = "keyboard",
) -> None:
    print(f"Quit key: `{quit_key}`")
    if controller == "vr":
        print(
            "VR controls: left thumbstick right start/stop and save, "
            "left thumbstick left discard, left thumbstick up save and quit, "
            "left thumbstick down reset robot pose"
        )
    if recording is None:
        return
    if controller == "keyboard":
        print(f"Recording hotkeys: `{start_key}` start, `{stop_key}` stop and save")


def _load_example_module(file_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to build import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_vr_events_to_recording_controls(vr_events: dict[str, bool] | None) -> VRRecordingControls:
    if not vr_events:
        return VRRecordingControls()

    return VRRecordingControls(
        toggle_recording=bool(vr_events.get("exit_early")),
        discard_episode=bool(vr_events.get("rerecord_episode")),
        quit_session=bool(vr_events.get("stop_recording")),
        reset_robot=bool(vr_events.get("reset_position")),
    )


def _decide_vr_recording_action(active: bool, controls: VRRecordingControls) -> VRRecordingDecision:
    toggle_requested = (
        controls.toggle_recording
        and not controls.discard_episode
        and not controls.quit_session
    )
    return VRRecordingDecision(
        start_recording=toggle_requested and not active,
        save_episode=toggle_requested and active,
        discard_episode=controls.discard_episode and active,
        quit_session=controls.quit_session,
        reset_robot=controls.reset_robot,
    )


def _apply_vr_recording_decision(
    recording: RecordingSession,
    decision: VRRecordingDecision,
) -> None:
    if decision.start_recording:
        recording.active = True
        print("Recording started from VR. Push left thumbstick right again to save.")
    if decision.save_episode:
        _save_episode(recording)
    if decision.discard_episode:
        _discard_episode(recording)
    if decision.quit_session:
        print("Stopping the VR recording session.")


if __name__ == "__main__":
    raise SystemExit(main())
