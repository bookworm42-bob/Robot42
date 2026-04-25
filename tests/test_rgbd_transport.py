from __future__ import annotations

import unittest

from xlerobot_playground.rgbd_transport import pack_rgbd_frame, unpack_rgbd_frame


class RgbdTransportTests(unittest.TestCase):
    def test_round_trip_paired_frame(self) -> None:
        payload = pack_rgbd_frame(
            frame_index=4,
            timestamp_us=1_500_000,
            rgb=b"abcdef",
            rgb_width=1,
            rgb_height=2,
            depth_be=(1000).to_bytes(2, "big") + (2000).to_bytes(2, "big"),
            depth_width=1,
            depth_height=2,
        )

        frame = unpack_rgbd_frame(payload)

        self.assertEqual(frame.frame_index, 4)
        self.assertAlmostEqual(frame.timestamp_s, 1.5)
        self.assertEqual(frame.rgb, b"abcdef")
        self.assertEqual(frame.depth_be, (1000).to_bytes(2, "big") + (2000).to_bytes(2, "big"))

    def test_pack_rejects_mismatched_rgb_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB payload size"):
            pack_rgbd_frame(
                frame_index=1,
                timestamp_us=1,
                rgb=b"abc",
                rgb_width=2,
                rgb_height=1,
            )

    def test_unpack_rejects_trailing_bytes(self) -> None:
        payload = pack_rgbd_frame(
            frame_index=1,
            timestamp_us=1,
            rgb=b"abc",
            rgb_width=1,
            rgb_height=1,
        )

        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            unpack_rgbd_frame(payload + b"x")


if __name__ == "__main__":
    unittest.main()
