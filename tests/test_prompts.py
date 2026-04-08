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
        self.assertIn("validated map-level frontier candidates", prompt)
        self.assertIn("sensor-range edge frontiers", prompt)
        self.assertIn("frontier_ids_to_store", prompt)

    def test_exploration_policy_user_prompt_includes_ascii_map_and_guardrails(self) -> None:
        prompt = build_exploration_policy_user_prompt(
            {
                "mission": "Explore the apartment.",
                "robot": {"coverage": 0.4},
                "frontier_memory": {"stored_frontiers": []},
                "candidate_frontiers": [{"frontier_id": "frontier_001"}],
                "explored_areas": [{"region_id": "region_hallway"}],
                "recent_views": [{"frame_id": "kf_001"}],
                "guardrails": {"finish_requires_frontier_exhaustion": True},
                "ascii_map": "??\nRF",
            }
        )
        self.assertIn("ASCII Occupancy Map:", prompt)
        self.assertIn("finish_requires_frontier_exhaustion", prompt)
        self.assertIn("RF", prompt)


if __name__ == "__main__":
    unittest.main()
