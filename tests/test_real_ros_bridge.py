from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from xlerobot_playground.real_ros_bridge import (
    OrbbecFilesystemConfig,
    OrbbecFilesystemRgbdSource,
    RobotBrainRgbdSource,
    _motion_result_error,
    build_parser,
    config_from_args,
    _format_runtime_error,
    imu_ros_timestamp_s,
    parse_imu_json,
    parse_depth_pgm_mm,
    parse_rgb_ppm,
    read_depth_pgm_mm,
    synthesize_scan_from_depth_rows,
    twist_to_base_velocity,
    yaw_to_quaternion_xyzw,
)


class _Vector:
    def __init__(self, *, x: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.z = z


class _Twist:
    def __init__(self) -> None:
        self.linear = _Vector(x=0.04)
        self.angular = _Vector(z=0.12)


class _FakeBrainClient:
    def __init__(self) -> None:
        self.requested_paths: list[str] = []
        self.payloads = {
            "/rgb": b"P6\n1 1\n255\nabc",
            "/depth": b"P5\n1 1\n65535\n" + (1234).to_bytes(2, "big"),
        }

    def get_bytes(self, path: str) -> bytes:
        self.requested_paths.append(path)
        return self.payloads[path]


class RealRosBridgeTests(unittest.TestCase):
    def test_parser_defaults_match_two_wheel_robot_ports(self) -> None:
        args = build_parser().parse_args([])
        config = config_from_args(args)

        self.assertEqual(config.robot_kind, "xlerobot_2wheels")
        self.assertEqual(config.port1, "/dev/tty.usbmodem5B140330101")
        self.assertEqual(config.port2, "/dev/tty.usbmodem5B140332271")
        self.assertFalse(config.allow_motion_commands)
        self.assertEqual(config.odom_source, "none")
        self.assertEqual(config.max_linear_m_s, 0.05)
        self.assertEqual(config.max_angular_rad_s, 0.20)
        self.assertEqual(config.camera_z_m, 0.35)

    def test_camera_mount_arguments_are_configurable(self) -> None:
        args = build_parser().parse_args(
            ["--camera-x-m", "0.04", "--camera-y-m", "-0.01", "--camera-z-m", "0.32", "--camera-yaw-rad", "0.1"]
        )
        config = config_from_args(args)

        self.assertEqual(config.camera_x_m, 0.04)
        self.assertEqual(config.camera_y_m, -0.01)
        self.assertEqual(config.camera_z_m, 0.32)
        self.assertEqual(config.camera_yaw_rad, 0.1)

    def test_commanded_odom_is_explicit_smoke_test_mode(self) -> None:
        args = build_parser().parse_args(["--odom-source", "commanded"])
        config = config_from_args(args)

        self.assertEqual(config.odom_source, "commanded")

    def test_robot_brain_url_selects_remote_hardware_endpoint(self) -> None:
        args = build_parser().parse_args(
            [
                "--robot-brain-url",
                "http://robot-brain.local:8765",
                "--imu-ws-path",
                "/ws/imu",
                "--imu-ws-reconnect-delay-s",
                "0.5",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.robot_brain_url, "http://robot-brain.local:8765")
        self.assertEqual(config.imu_topic, "/imu")
        self.assertEqual(config.imu_ws_path, "/ws/imu")
        self.assertEqual(config.imu_ws_reconnect_delay_s, 0.5)

    def test_runtime_error_formatter_includes_robot_brain_http_body(self) -> None:
        exc = HTTPError(
            url="http://robot-brain.local:8765/cmd_vel",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b"(6, 'Device not configured')"),
        )

        self.assertEqual(_format_runtime_error(exc), "HTTP 500: (6, 'Device not configured')")

    def test_motion_result_error_reports_rejected_remote_command(self) -> None:
        self.assertEqual(
            _motion_result_error(
                {
                    "succeeded": False,
                    "message": "Real motion commands are disabled.",
                    "metadata": {"requested_angular_rad_s": 0.3},
                }
            ),
            "Real motion commands are disabled. metadata={'requested_angular_rad_s': 0.3}",
        )
        self.assertIsNone(_motion_result_error({"succeeded": True, "message": "ok"}))

    def test_twist_to_base_velocity_uses_forward_and_yaw_only(self) -> None:
        self.assertEqual(twist_to_base_velocity(_Twist()), (0.04, 0.12))

    def test_yaw_to_quaternion(self) -> None:
        _x, _y, z, w = yaw_to_quaternion_xyzw(0.0)

        self.assertEqual(z, 0.0)
        self.assertEqual(w, 1.0)

    def test_depth_rows_convert_to_scan_ranges(self) -> None:
        depth = tuple(tuple(1000 for _ in range(5)) for _ in range(7))

        ranges, angles = synthesize_scan_from_depth_rows(
            depth,
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=4.0,
        )

        self.assertEqual(len(ranges), 5)
        self.assertEqual(len(angles), 5)
        self.assertTrue(all(value >= 1.0 for value in ranges[1:4]))
        self.assertLess(angles[0], 0.0)
        self.assertGreater(angles[-1], 0.0)

    def test_depth_rows_fill_no_return_beams(self) -> None:
        depth = tuple(
            tuple(1000 if column == 2 else 0 for column in range(5))
            for _row in range(7)
        )

        ranges, _angles = synthesize_scan_from_depth_rows(
            depth,
            horizontal_fov_rad=1.0,
            band_height_px=3,
            range_min_m=0.05,
            range_max_m=4.0,
        )

        self.assertEqual(ranges[0], 4.0)
        self.assertLess(ranges[2], 1.1)
        self.assertEqual(ranges[-1], 4.0)

    def test_reads_16_bit_depth_pgm_as_millimetres(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_depth.pgm"
            path.write_bytes(
                b"P5\n2 2\n65535\n"
                + (1000).to_bytes(2, "big")
                + (2000).to_bytes(2, "big")
                + (0).to_bytes(2, "big")
                + (65535).to_bytes(2, "big")
            )

            depth, width, height = read_depth_pgm_mm(path)

        self.assertEqual((width, height), (2, 2))
        self.assertEqual(depth[0], (1000, 2000))
        self.assertEqual(depth[1], (0, 65535))

    def test_parse_pnm_payloads_from_robot_brain(self) -> None:
        rgb, rgb_width, rgb_height = parse_rgb_ppm(b"P6\n1 1\n255\nabc")
        depth, depth_width, depth_height = parse_depth_pgm_mm(
            b"P5\n1 1\n65535\n" + (1234).to_bytes(2, "big")
        )

        self.assertEqual(rgb, b"abc")
        self.assertEqual((rgb_width, rgb_height), (1, 1))
        self.assertEqual(depth, ((1234,),))
        self.assertEqual((depth_width, depth_height), (1, 1))

    def test_parse_imu_json_reads_nested_metadata_contract(self) -> None:
        sample = parse_imu_json(
            b'{"imu":{"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0},"system_timestamp_us":1234567}}'
        )

        self.assertEqual(sample["angular_velocity_rad_s"]["z"], 0.3)
        self.assertEqual(sample["linear_acceleration_m_s2"]["x"], 1.0)
        self.assertAlmostEqual(sample["timestamp_s"], 1.234567)

    def test_imu_ros_timestamp_prefers_accel_frame_time(self) -> None:
        sample = parse_imu_json(
            b'{"timestamp_s":2.0,"has_accel":true,"has_gyro":true,'
            b'"accel_timestamp_us":1500000,"gyro_timestamp_us":2000000,'
            b'"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},'
            b'"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0}}'
        )

        self.assertAlmostEqual(imu_ros_timestamp_s(sample), 1.5)

    def test_imu_ros_timestamp_keeps_gyro_time_for_gyro_only_samples(self) -> None:
        sample = parse_imu_json(
            b'{"timestamp_s":2.0,"has_accel":false,"has_gyro":true,'
            b'"gyro_timestamp_us":2000000,'
            b'"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3}}'
        )

        self.assertAlmostEqual(imu_ros_timestamp_s(sample), 2.0)

    def test_filesystem_source_can_return_rgb_without_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "latest.ppm").write_bytes(b"P6\n1 1\n255\nabc")
            source = OrbbecFilesystemRgbdSource(OrbbecFilesystemConfig(output_dir=output_dir))

            frame = source.capture()

        self.assertEqual(frame.rgb, b"abc")
        self.assertEqual(frame.rgb_width, 1)
        self.assertEqual(frame.rgb_height, 1)
        self.assertIsNone(frame.depth_mm)

    def test_robot_brain_rgbd_source_reads_remote_pnm_payloads(self) -> None:
        client = _FakeBrainClient()
        source = RobotBrainRgbdSource(client)

        frame = source.capture()

        self.assertEqual(frame.rgb, b"abc")
        self.assertEqual(frame.rgb_width, 1)
        self.assertEqual(frame.depth_mm, ((1234,),))
        self.assertEqual(frame.depth_width, 1)
        self.assertIsNone(frame.imu_sample)
        self.assertEqual(client.requested_paths, ["/rgb", "/depth"])


if __name__ == "__main__":
    unittest.main()
