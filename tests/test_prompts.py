import unittest

from xlerobot_agent.prompts import (
    build_action_selection_system_prompt,
    build_exploration_policy_system_prompt,
    build_exploration_policy_user_prompt,
    build_skill_selection_system_prompt,
    build_subgoal_planning_system_prompt,
)


class PromptExamplesTests(unittest.TestCase):
    def test_action_selection_prompt_mentions_navigation_and_perception_patterns(self) -> None:
        prompt = build_action_selection_system_prompt()
        self.assertIn("get_map -> create_map/explore if needed -> go_to_pose", prompt)
        self.assertIn("perceive_scene -> ground_object_3d -> set_waypoint_from_object -> go_to_pose", prompt)
        self.assertIn("run_vla_skill::<skill_id>", prompt)

    def test_subgoal_planning_prompt_uses_examples_without_tool_ids_as_subgoals(self) -> None:
        prompt = build_subgoal_planning_system_prompt()
        self.assertIn("navigate(kitchen) -> search(fridge) -> align(fridge) -> manipulate(fridge)", prompt)
        self.assertIn("Do not emit tool names like `get_map` or `go_to_pose`", prompt)

    def test_skill_selection_prompt_mentions_tool_then_skill_pattern(self) -> None:
        prompt = build_skill_selection_system_prompt()
        self.assertIn("Prefer skills for acting on the world", prompt)
        self.assertIn("prefer tools first when the agent still needs map state", prompt)

    def test_exploration_policy_prompt_mentions_frontier_memory_and_sensor_range(self) -> None:
        prompt = build_exploration_policy_system_prompt()
        self.assertIn("frontier information", prompt)
        self.assertIn("partial RGB-D-derived evidence", prompt)
        self.assertIn("recent RGB visual input", prompt)
        self.assertIn("coordinates at boundaries", prompt)
        self.assertIn("robot-navigable floor space", prompt)
        self.assertIn("couches", prompt)
        self.assertIn("sensor-range edge frontiers", prompt)
        self.assertIn("frontier_ids_to_store", prompt)
        self.assertIn("memory_updates", prompt)
        self.assertIn("same response", prompt)
        self.assertIn("veto", prompt)
        self.assertIn("free_space_path_distance_m", prompt)
        self.assertIn("Locality is a primary objective", prompt)
        self.assertIn("nearest useful region", prompt)
        self.assertIn("2x farther", prompt)
        self.assertIn("avoid zig-zagging", prompt)

    def test_exploration_policy_user_prompt_includes_ascii_map_and_guardrails(self) -> None:
        prompt = build_exploration_policy_user_prompt(
            {
                "mission": "Explore the apartment.",
                "robot": {"coverage": 0.4},
                "frontier_memory": {"stored_frontiers": []},
                "frontier_information": [
                    {
                        "frontier_id": "frontier_001",
                        "currently_visible": True,
                        "free_space_path_distance_m": 1.25,
                    }
                ],
                "frontier_selection_guidance": ["Select likely navigable openings."],
                "navigation_map_views": [
                    {
                        "frame_id": "nav_map_001",
                        "thumbnail_data_url": "data:image/png;base64," + "b" * 128,
                    }
                ],
                "explored_areas": [{"region_id": "region_hallway"}],
                "recent_views": [{"frame_id": "kf_001"}],
                "guardrails": {"finish_requires_frontier_exhaustion": True},
                "ascii_map": "??\nRF",
            }
        )
        self.assertIn("ASCII Occupancy Map:", prompt)
        self.assertIn("Frontier Information:", prompt)
        self.assertIn("Frontier Memory Status:", prompt)
        self.assertIn("latest scan/global-map update", prompt)
        self.assertIn("source", prompt)
        self.assertIn("current_rgbd_scan", prompt)
        self.assertIn("frontier_memory", prompt)
        self.assertIn("previous scans", prompt)
        self.assertIn("free_space_path_distance_m", prompt)
        self.assertIn("known free space", prompt)
        self.assertIn("Locality is a primary objective", prompt)
        self.assertIn("2x farther", prompt)
        self.assertIn("avoid jumping across the apartment", prompt)
        self.assertIn("Navigation Map View:", prompt)
        self.assertIn("nav_map_001", prompt)
        self.assertIn("recent RGB visual input", prompt)
        self.assertIn("Memory Update Instructions:", prompt)
        self.assertIn("Select likely navigable openings.", prompt)
        self.assertIn("finish_requires_frontier_exhaustion", prompt)
        self.assertIn("RF", prompt)
        self.assertIn("veto", prompt)

    def test_exploration_policy_user_prompt_compacts_frontier_memory_details(self) -> None:
        prompt = build_exploration_policy_user_prompt(
            {
                "frontier_memory": {
                    "stored_frontiers": [
                        {
                            "frontier_id": "frontier_999",
                            "status": "stored",
                            "discovered_step": 2,
                            "last_seen_step": 2,
                            "approach_pose": {"x": 9.0, "y": 9.0, "yaw": 0.0},
                            "evidence": ["memory duplicate evidence should not be repeated"],
                        }
                    ],
                    "return_waypoints": [],
                },
                "frontier_information": [
                    {
                        "frontier_id": "frontier_001",
                        "status": "stored",
                        "currently_visible": True,
                        "path_cost_m": 2.0,
                        "approach_pose": {"x": 1.0, "y": 1.0, "yaw": 0.0},
                        "frontier_boundary_pose": {"x": 1.2, "y": 1.1, "yaw": 0.0},
                        "evidence": ["canonical frontier evidence"],
                    }
                ],
            }
        )

        self.assertIn("frontier_999", prompt)
        self.assertIn('"source": "frontier_memory"', prompt)
        self.assertIn('"source": "current_rgbd_scan"', prompt)
        self.assertIn('"free_space_path_distance_m": 2.0', prompt)
        self.assertIn("canonical frontier evidence", prompt)
        self.assertNotIn("memory duplicate evidence should not be repeated", prompt)
        self.assertNotIn("currently_visible", prompt)
        self.assertNotIn('"x": 9.0', prompt)

    def test_exploration_policy_user_prompt_redacts_data_urls(self) -> None:
        prompt = build_exploration_policy_user_prompt(
            {
                "recent_views": [
                    {
                        "frame_id": "kf_001",
                        "thumbnail_data_url": "data:image/png;base64," + "a" * 128,
                    }
                ]
            }
        )
        self.assertIn("attached_as_multimodal_image", prompt)
        self.assertIn("base64_bytes", prompt)
        self.assertNotIn("a" * 64, prompt)


if __name__ == "__main__":
    unittest.main()
