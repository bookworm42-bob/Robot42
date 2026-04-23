from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xlerobot_playground.robot_brain_agent import (
    RobotBrainAgent,
    RobotBrainAgentConfig,
    build_parser,
    config_from_args,
)


class FakeResult:
    def __init__(self, succeeded=True, message="ok", metadata=None) -> None:
        self.succeeded = succeeded
        self.message = message
        self.metadata = metadata or {}


class FakeRuntime:
    def __init__(self) -> None:
        self.velocity_calls = []
        self.stop_calls = 0
        self.close_calls = 0

    def drive_velocity(self, *, linear_m_s: float, angular_rad_s: float):
        self.velocity_calls.append((linear_m_s, angular_rad_s))
        return FakeResult(metadata={"sent": True})

    def stop(self):
        self.stop_calls += 1
        return FakeResult(message="stopped")

    def close(self) -> None:
        self.close_calls += 1


class RobotBrainAgentTests(unittest.TestCase):
    def test_parser_defaults_match_robot_brain_deployment(self) -> None:
        args = build_parser().parse_args([])
        config = config_from_args(args)

        self.assertEqual(config.robot_kind, "xlerobot_2wheels")
        self.assertEqual(config.port1, "/dev/tty.usbmodem5B140330101")
        self.assertEqual(config.port2, "/dev/tty.usbmodem5B140332271")
        self.assertFalse(config.allow_motion_commands)
        self.assertEqual(config.port, 8765)
        self.assertFalse(config.debug_motion)
        self.assertEqual(config.calibration_prompt_response, "")
        self.assertEqual(config.imu_filename, "latest_imu.json")

    def test_parser_accepts_debug_motion(self) -> None:
        args = build_parser().parse_args(["--debug-motion"])
        config = config_from_args(args)

        self.assertTrue(config.debug_motion)

    def test_parser_accepts_interactive_calibration(self) -> None:
        args = build_parser().parse_args(["--interactive-calibration"])
        config = config_from_args(args)

        self.assertIsNone(config.calibration_prompt_response)

    def test_agent_forwards_velocity_to_runtime(self) -> None:
        runtime = FakeRuntime()
        agent = RobotBrainAgent(RobotBrainAgentConfig(), runtime=runtime)

        response = agent.velocity(linear_m_s=0.02, angular_rad_s=0.08)

        self.assertTrue(response["succeeded"])
        self.assertEqual(runtime.velocity_calls, [(0.02, 0.08)])
        self.assertEqual(response["metadata"], {"sent": True})

    def test_agent_serves_expected_orbbec_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = RobotBrainAgent(
                RobotBrainAgentConfig(orbbec_output_dir=Path(tmpdir)),
                runtime=FakeRuntime(),
            )

            self.assertEqual(agent.file_path("/rgb"), Path(tmpdir) / "latest.ppm")
            self.assertEqual(agent.file_path("/depth"), Path(tmpdir) / "latest_depth.pgm")
            self.assertEqual(agent.file_path("/metadata"), Path(tmpdir) / "latest.json")
            self.assertEqual(agent.file_path("/imu"), Path(tmpdir) / "latest_imu.json")
            self.assertIsNone(agent.file_path("/missing"))


if __name__ == "__main__":
    unittest.main()
