from __future__ import annotations

import unittest

from xlerobot_playground.imu_transport import build_websocket_url, parse_imu_json


class ImuTransportTests(unittest.TestCase):
    def test_parse_imu_json_keeps_explicit_fields(self) -> None:
        sample = parse_imu_json(
            b'{"timestamp_s":1.5,"angular_velocity_rad_s":{"x":0.1,"y":0.2,"z":0.3},"linear_acceleration_m_s2":{"x":1.0,"y":2.0,"z":3.0},"accel_frame_index":11,"gyro_frame_index":17,"accel_temperature_c":25.0,"gyro_temperature_c":26.0}'
        )

        self.assertEqual(sample["accel_frame_index"], 11)
        self.assertEqual(sample["gyro_frame_index"], 17)
        self.assertEqual(sample["gyro_temperature_c"], 26.0)

    def test_build_websocket_url_swaps_scheme(self) -> None:
        self.assertEqual(
            build_websocket_url("http://robot-brain.local:8765", "/ws/imu"),
            "ws://robot-brain.local:8765/ws/imu",
        )
        self.assertEqual(
            build_websocket_url("https://robot-brain.local", "/ws/imu"),
            "wss://robot-brain.local/ws/imu",
        )


if __name__ == "__main__":
    unittest.main()
