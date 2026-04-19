from __future__ import annotations

import unittest

from xlerobot_playground.real_agentic_exploration import build_parser, translated_args


class RealAgenticExplorationTests(unittest.TestCase):
    def test_defaults_translate_to_ros_nav2_real_exploration(self) -> None:
        args = build_parser().parse_args([])

        translated = translated_args(args)

        self.assertIn("--nav2-mode", translated)
        self.assertEqual(translated[translated.index("--nav2-mode") + 1], "ros")
        self.assertIn("--serve-review-ui", translated)
        self.assertIn("--ros-navigation-map-source", translated)
        self.assertEqual(translated[translated.index("--ros-navigation-map-source") + 1], "fused_scan")
        self.assertEqual(translated[translated.index("--source") + 1], "real_xlerobot")
        self.assertEqual(translated[translated.index("--ros-manual-spin-angular-speed-rad-s") + 1], "0.1")

    def test_explicit_llm_and_ui_options_are_preserved(self) -> None:
        args = build_parser().parse_args(
            [
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-test",
                "--llm-api-key",
                "secret",
                "--no-serve-review-ui",
                "--review-host",
                "127.0.0.1",
                "--review-port",
                "8899",
            ]
        )

        translated = translated_args(args)

        self.assertEqual(translated[translated.index("--llm-provider") + 1], "openai")
        self.assertEqual(translated[translated.index("--llm-model") + 1], "gpt-test")
        self.assertEqual(translated[translated.index("--llm-api-key") + 1], "secret")
        self.assertIn("--no-serve-review-ui", translated)
        self.assertEqual(translated[translated.index("--review-host") + 1], "127.0.0.1")
        self.assertEqual(translated[translated.index("--review-port") + 1], "8899")


if __name__ == "__main__":
    unittest.main()
