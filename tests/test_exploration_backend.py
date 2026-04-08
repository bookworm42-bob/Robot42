from __future__ import annotations

import tempfile
import unittest

from xlerobot_agent.exploration import ExplorationBackend, ExplorationBackendConfig


class ExplorationBackendExternalTaskTests(unittest.TestCase):
    def test_external_task_updates_and_completes_with_named_places(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ExplorationBackend(
                ExplorationBackendConfig(
                    mode="sim",
                    persist_path=f"{tmpdir}/map.json",
                    occupancy_resolution=0.25,
                )
            )
            task = backend.begin_external_task(
                tool_id="explore",
                area="workspace",
                session="house_v1",
                source="operator",
            )
            backend.update_external_task(
                task["task_id"],
                progress=0.4,
                message="Exploring",
                result={"trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}]},
            )
            map_payload = {
                "map_id": "house_v1",
                "frame": "map",
                "resolution": 0.25,
                "coverage": 12.0,
                "summary": "test map",
                "approved": False,
                "created_at": 1.0,
                "source": "operator",
                "mode": "sim",
                "trajectory": [{"x": 0.0, "y": 0.0, "yaw": 0.0}],
                "keyframes": [],
                "regions": [
                    {
                        "region_id": "region_01",
                        "label": "kitchen",
                        "confidence": 0.8,
                        "polygon_2d": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                        "centroid": {"x": 1.0, "y": 1.0},
                        "adjacency": [],
                        "representative_keyframes": [],
                        "evidence": ["fridge visible"],
                        "default_waypoints": [{"name": "kitchen_center", "x": 1.0, "y": 1.0, "yaw": 0.0}],
                    }
                ],
                "named_places": [],
                "occupancy": {
                    "resolution": 0.25,
                    "bounds": {"min_x": 0.0, "max_x": 2.0, "min_y": 0.0, "max_y": 2.0},
                    "cells": [{"x": 0.0, "y": 0.0, "state": "free"}],
                },
            }
            backend.complete_external_task(task["task_id"], map_payload=map_payload)

            snapshot = backend.snapshot()
            self.assertEqual(snapshot["active_task"]["state"], "succeeded")
            self.assertEqual(snapshot["current_map"]["map_id"], "house_v1")
            named_places = {item["name"] for item in snapshot["current_map"]["named_places"]}
            self.assertIn("kitchen_entry", named_places)
            self.assertIn("kitchen_center", named_places)


if __name__ == "__main__":
    unittest.main()
