from __future__ import annotations

import unittest

from unittest.mock import patch

from xlerobot_playground.ros_pitched_forward_diagnostic import build_forward_args, build_parser, post_camera_pitch


class RosPitchedForwardDiagnosticTests(unittest.TestCase):
    def test_build_forward_args_passes_pitch_test_defaults(self) -> None:
        args = build_parser().parse_args([])

        forward_args = build_forward_args(args)

        self.assertIn("--send-motion", forward_args)
        self.assertEqual(forward_args[forward_args.index("--target-distance-m") + 1], "1.0")
        self.assertEqual(forward_args[forward_args.index("--imu-topic") + 1], "/imu/filtered_yaw")
        self.assertEqual(forward_args[forward_args.index("--imu-frame-convention") + 1], "base_link")
        self.assertEqual(
            forward_args[forward_args.index("--json-out") + 1],
            "artifacts/diagnostics/forward_pitch35_1m_summary.json",
        )

    def test_post_camera_pitch_uses_physical_pitch_endpoint(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b'{"succeeded":true}'

        with patch("xlerobot_playground.ros_pitched_forward_diagnostic.request.urlopen", return_value=_Response()) as mocked:
            response = post_camera_pitch(
                robot_brain_url="http://brain:8765",
                pitch_deg=35.0,
                action_key="head_tilt.pos",
                settle_s=0.1,
            )

        self.assertTrue(response["succeeded"])
        req = mocked.call_args.args[0]
        self.assertEqual(req.full_url, "http://brain:8765/camera/head/pitch")
        self.assertIn(b'"action_key": "head_tilt.pos"', req.data)


if __name__ == "__main__":
    unittest.main()
