import unittest

from xlerobot_agent import (
    DelegatedNavigationBackend,
    GoalContext,
    MockPromptClient,
    MockVoiceTranslator,
    MockWakeWordDetector,
    NavigationSkillExecutionMode,
    PromptPlanner,
    SkillContract,
    SkillRegistry,
    SkillType,
    VoiceCommandPipeline,
    WakeWordConfig,
    WorldState,
    XLeRobotAgentRuntime,
    build_default_navigation_skills,
    create_executor_config,
)
from xlerobot_agent.integration import XLeRobotAgentBindings, default_executor_registry


class XLeRobotAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bindings = XLeRobotAgentBindings()
        self.executor_config = create_executor_config(
            NavigationSkillExecutionMode.VLA_NAVIGATION_SKILLS
        )
        self.registry = SkillRegistry()
        self.registry.register_many(build_default_navigation_skills(self.executor_config, self.bindings))
        self.registry.register(
            SkillContract(
                skill_id="open_fridge",
                skill_type=SkillType.MANIPULATION,
                language_description="Open the fridge door.",
                executor_binding=self.bindings.generic_skill_binding,
                required_resources=frozenset({"left_arm", "right_arm"}),
                required_observations=frozenset({"fridge_handle_visible"}),
                expected_postcondition="fridge is open",
                value_function_id="open_fridge_success",
                min_localization_confidence=0.55,
            )
        )
        self.runtime = XLeRobotAgentRuntime(
            registry=self.registry,
            planner=PromptPlanner(prompt_client=MockPromptClient()),
            executors=default_executor_registry(self.executor_config, self.bindings),
        )

    def test_navigation_skill_is_selected_when_manipulation_is_not_feasible(self) -> None:
        world_state = WorldState(
            current_task="Go to the kitchen and open the fridge.",
            localization_confidence=0.9,
            affordance_predictions={
                "navigate_to_region": 0.9,
                "open_fridge": 0.9,
            },
            available_observations=frozenset(),
        )
        record = self.runtime.run_instruction("Go to the kitchen and open the fridge.", world_state)
        self.assertTrue(record.steps)
        self.assertEqual(record.steps[0].selected_skill.skill_id, "navigate_to_region")

    def test_low_localization_blocks_navigation_skill(self) -> None:
        world_state = WorldState(
            current_task="Navigate to the kitchen.",
            localization_confidence=0.1,
            affordance_predictions={
                "navigate_to_region": 0.95,
            },
        )
        planner = PromptPlanner(prompt_client=MockPromptClient())
        goal = GoalContext("Navigate to the kitchen.", "Navigate to the kitchen.")
        subgoal = planner.plan_subgoals(
            goal,
            world_state,
            self.registry.list_enabled(),
        )[0]
        scores = planner.score_skills_for_subgoal(
            goal,
            subgoal,
            world_state,
            self.registry.list_enabled(),
        )
        nav_score = next(score for score in scores if score.skill_id == "navigate_to_region")
        self.assertFalse(nav_score.feasible)
        self.assertIn("localization_below_threshold", nav_score.rejection_reasons)

    def test_delegated_backend_changes_navigation_binding(self) -> None:
        config = create_executor_config(
            NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE,
            DelegatedNavigationBackend.PROGRESSIVE_MAP,
        )
        skills = build_default_navigation_skills(config, self.bindings)
        self.assertEqual(
            skills[0].executor_binding,
            self.bindings.delegated_progressive_map_binding,
        )

    def test_voice_pipeline_requires_wake_word(self) -> None:
        pipeline = VoiceCommandPipeline(
            wake_word_detector=MockWakeWordDetector(WakeWordConfig(wake_word="hey xlerobot")),
            translator=MockVoiceTranslator(),
        )
        self.assertIsNone(pipeline.process_transcript("open the fridge"))
        command = pipeline.process_transcript("hey xlerobot open the fridge")
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command.normalized_command, "open the fridge")

    def test_place_discovery_and_subgoal_decomposition(self) -> None:
        world_state = WorldState(
            current_task="Open the fridge.",
            current_pose="hallway",
            visible_objects=frozenset({"fridge"}),
            image_descriptions=("a kitchen area with a fridge",),
            localization_confidence=0.8,
        )
        record = self.runtime.run_instruction("Go to the kitchen and open the fridge.", world_state)
        self.assertIn("kitchen", {place.name for place in record.discovered_places})
        self.assertTrue(any("kitchen" in subgoal.text for subgoal in record.subgoals))


if __name__ == "__main__":
    unittest.main()
