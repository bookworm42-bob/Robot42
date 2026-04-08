import unittest

from xlerobot_playground.real_backend import (
    VRRecordingControls,
    _decide_vr_recording_action,
    _map_vr_events_to_recording_controls,
)


class RealBackendVRControlTests(unittest.TestCase):
    def test_exit_early_starts_recording_when_idle(self) -> None:
        controls = _map_vr_events_to_recording_controls({"exit_early": True})
        decision = _decide_vr_recording_action(False, controls)
        self.assertEqual(
            decision.start_recording,
            True,
        )
        self.assertEqual(decision.save_episode, False)

    def test_exit_early_saves_episode_when_active(self) -> None:
        controls = _map_vr_events_to_recording_controls({"exit_early": True})
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.start_recording, False)
        self.assertEqual(decision.save_episode, True)

    def test_rerecord_takes_precedence_over_toggle(self) -> None:
        controls = _map_vr_events_to_recording_controls(
            {
                "exit_early": True,
                "rerecord_episode": True,
            }
        )
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.save_episode, False)
        self.assertEqual(decision.discard_episode, True)

    def test_stop_recording_takes_precedence_over_toggle(self) -> None:
        controls = _map_vr_events_to_recording_controls(
            {
                "exit_early": True,
                "stop_recording": True,
            }
        )
        decision = _decide_vr_recording_action(True, controls)
        self.assertEqual(decision.save_episode, False)
        self.assertEqual(decision.quit_session, True)

    def test_reset_position_maps_to_reset_robot(self) -> None:
        controls = _map_vr_events_to_recording_controls({"reset_position": True})
        self.assertEqual(
            controls,
            VRRecordingControls(reset_robot=True),
        )


if __name__ == "__main__":
    unittest.main()
