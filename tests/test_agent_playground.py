import unittest
from unittest.mock import patch
import time

from xlerobot_agent import GoalContext, NavigationSkillExecutionMode, Subgoal, WorldState
from xlerobot_agent.environment import build_environment_adapter
from xlerobot_agent.llm import AgentLLMRouter, AgentModelSuite, LLMCallTrace, ModelConfig
from xlerobot_agent.playground import PlaygroundAgentRuntime
from xlerobot_agent.reporting import LiveAgentReport
from xlerobot_agent.tools import ToolCallContext, build_default_tool_registry
from xlerobot_agent.visual_differencing import VisualDifferencingModule


class PlaygroundAgentRuntimeTests(unittest.TestCase):
    def _make_runtime(self, world_state: WorldState) -> PlaygroundAgentRuntime:
        adapter = build_environment_adapter(
            backend="sim",
            initial_world_state=world_state,
            navigation_mode=NavigationSkillExecutionMode.DELEGATED_NAVIGATION_MODULE,
        )
        model_suite = AgentModelSuite(
            planner=ModelConfig(provider="mock", model="mock-planner"),
            critic=ModelConfig(provider="mock", model="mock-critic"),
            coder=ModelConfig(provider="mock", model="mock-coder"),
        )
        report = LiveAgentReport(
            backend="sim",
            models={role: (config.__dict__ if config is not None else None) for role, config in model_suite.__dict__.items()},
            environment=adapter.describe_environment(),
        )
        return PlaygroundAgentRuntime(
            environment=adapter,
            llm_router=AgentLLMRouter(model_suite),
            tools=build_default_tool_registry(),
            report=report,
        )

    def test_playground_runtime_completes_simple_command(self) -> None:
        runtime = self._make_runtime(
            WorldState(
                current_task="Go to the kitchen and open the fridge.",
                current_pose="hallway",
                localization_confidence=0.9,
                visible_objects=frozenset({"counter"}),
                image_descriptions=("a hallway leading toward a kitchen",),
                affordance_predictions={
                    "navigate_to_region": 0.95,
                    "search_for_target": 0.85,
                    "open_fridge": 0.8,
                },
            )
        )
        record = runtime.run_instruction("Go to the kitchen and open the fridge.")
        self.assertEqual(record.status, "completed")
        snapshot = runtime.report.snapshot()
        kinds = [event["kind"] for event in snapshot["events"]]
        self.assertIn("plan_created", kinds)
        self.assertIn("llm_decision", kinds)
        self.assertIn("skill_executed", kinds)

    def test_playground_runtime_uses_code_execution_after_failures(self) -> None:
        runtime = self._make_runtime(
            WorldState(
                current_task="Open the fridge.",
                current_pose="kitchen",
                localization_confidence=0.9,
                visible_objects=frozenset(),
                image_descriptions=("a closed fridge is somewhere in the scene",),
                active_resource_locks=frozenset({"base", "head", "left_arm", "right_arm"}),
                affordance_predictions={
                    "search_for_target": 0.0,
                    "open_fridge": 0.0,
                    "align_for_skill": 0.0,
                    "navigate_to_region": 0.0,
                },
            )
        )
        record = runtime.run_instruction("Open the fridge.")
        self.assertEqual(record.status, "failed")
        snapshot = runtime.report.snapshot()
        kinds = [event["kind"] for event in snapshot["events"]]
        self.assertIn("code_generated", kinds)
        self.assertIn("tool_executed", kinds)

    def test_local_tool_registry_supports_perception_grounding(self) -> None:
        world_state = WorldState(
            current_task="Find the fridge.",
            current_pose="kitchen",
            visible_objects=frozenset({"fridge"}),
            image_descriptions=("a fridge door is visible near the counter",),
            metadata={
                "sensors": {"rgb": {"width": 640}, "depth": {"height": 480}},
                "object_anchors": {"fridge": {"x": 2.0, "y": 0.4, "z": 0.9, "depth_m": 2.0}},
            },
        )
        tools = build_default_tool_registry()
        context = ToolCallContext(
            goal=GoalContext("Find the fridge."),
            subgoal=Subgoal(text="find the fridge", kind="search", target="fridge"),
            world_state=world_state,
            payload={"target": "fridge"},
        )
        scene = tools.execute("perceive_scene", context)
        self.assertEqual(scene.status.value, "succeeded")
        self.assertTrue(scene.details["annotations"])
        self.assertIsNotNone(scene.updated_world_state)
        self.assertIn("fridge", scene.updated_world_state.visible_objects)

        grounding = tools.execute("ground_object_3d", context)
        self.assertEqual(grounding.status.value, "succeeded")
        self.assertEqual(grounding.details["best_match"]["label"], "fridge")

        waypoint = tools.execute("set_waypoint_from_object", context)
        self.assertEqual(waypoint.status.value, "succeeded")
        self.assertEqual(waypoint.recommended_action_id, "go_to_pose")

    def test_local_tool_registry_runs_mapping_jobs_and_resolves_region_navigation(self) -> None:
        world_state = WorldState(
            current_task="Map the home.",
            current_pose="hallway",
            localization_confidence=0.95,
        )
        tools = build_default_tool_registry()
        create_context = ToolCallContext(
            goal=GoalContext("Map the home."),
            subgoal=Subgoal(text="map the downstairs area", kind="general", target="downstairs"),
            world_state=world_state,
            payload={"session": "house_v1", "area": "downstairs"},
        )
        create_map = tools.execute("create_map", create_context)
        self.assertEqual(create_map.status.value, "in_progress")
        task_id = create_map.details["task"]["task_id"]

        deadline = time.time() + 5.0
        latest = None
        while time.time() < deadline:
            latest = tools.execute(
                "get_task_status",
                ToolCallContext(
                    goal=GoalContext("Map the home."),
                    subgoal=Subgoal(text="map the downstairs area", kind="general", target="downstairs"),
                    world_state=world_state,
                    payload={"task_id": task_id},
                ),
            )
            if latest.details["task"]["state"] == "succeeded":
                break
            time.sleep(0.05)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.details["task"]["state"], "succeeded")
        self.assertEqual(latest.recommended_action_id, "get_map")

        updated_world_state = latest.updated_world_state or world_state
        map_result = tools.execute(
            "get_map",
            ToolCallContext(
                goal=GoalContext("Go to the kitchen."),
                subgoal=Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"),
                world_state=updated_world_state,
                payload={"target": "kitchen"},
            ),
        )
        self.assertTrue(map_result.details["available"])
        self.assertEqual(map_result.recommended_action_id, "go_to_pose")

        go_to_pose = tools.execute(
            "go_to_pose",
            ToolCallContext(
                goal=GoalContext("Go to the kitchen."),
                subgoal=Subgoal(text="go to the kitchen", kind="navigate", target="kitchen"),
                world_state=map_result.updated_world_state or updated_world_state,
                payload={"pose": "kitchen"},
            ),
        )
        self.assertEqual(go_to_pose.status.value, "succeeded")
        self.assertEqual(go_to_pose.details["task"]["result"]["goal_label"], "kitchen")
        self.assertIsInstance(go_to_pose.details["task"]["result"]["goal_pose"], dict)

    def test_action_candidates_expose_vla_skill_wrapper_ids(self) -> None:
        runtime = self._make_runtime(
            WorldState(
                current_task="Open the fridge.",
                current_pose="kitchen",
                localization_confidence=0.9,
                visible_objects=frozenset({"fridge"}),
                image_descriptions=("a fridge is directly ahead",),
                available_observations=frozenset({"fridge_visible"}),
                affordance_predictions={
                    "open_fridge": 0.95,
                    "align_for_skill": 0.7,
                },
            )
        )
        goal = GoalContext("Open the fridge.", "Open the fridge.")
        subgoal = Subgoal(text="open the fridge", kind="manipulate", target="fridge")
        candidates = runtime._build_action_candidates(goal, subgoal, runtime.environment.build_world_state(), 0, None)
        wrapped = [item for item in candidates if item.action_type == "skill"]
        self.assertTrue(wrapped)
        self.assertTrue(all(item.action_id.startswith("run_vla_skill::") for item in wrapped))
        self.assertTrue(all(item.label == "run_vla_skill" for item in wrapped))

    def test_runtime_reports_visual_context_and_differences(self) -> None:
        runtime = self._make_runtime(
            WorldState(
                current_task="Go to the kitchen and open the fridge.",
                current_pose="hallway",
                localization_confidence=0.95,
                visible_objects=frozenset({"counter"}),
                image_descriptions=("a hallway leads toward the kitchen",),
                affordance_predictions={
                    "navigate_to_region": 0.95,
                    "align_for_skill": 0.8,
                    "open_fridge": 0.85,
                },
            )
        )
        record = runtime.run_instruction("Go to the kitchen and open the fridge.")
        self.assertEqual(record.status, "completed")
        snapshot = runtime.report.snapshot()
        kinds = [event["kind"] for event in snapshot["events"]]
        self.assertIn("visual_context", kinds)
        self.assertIn("visual_diff", kinds)
        diff_events = [event for event in snapshot["events"] if event["kind"] == "visual_diff"]
        self.assertTrue(diff_events)
        self.assertIn("vdm", diff_events[-1]["details"]["world_state"]["metadata"])

    def test_initial_visual_context_can_use_specialized_multimodal_model(self) -> None:
        model_suite = AgentModelSuite(
            planner=ModelConfig(provider="mock", model="mock-planner"),
            critic=ModelConfig(provider="mock", model="mock-critic"),
            coder=ModelConfig(provider="mock", model="mock-coder"),
            visual_summary=ModelConfig(
                provider="openai-compatible",
                model="vision-model",
                base_url="http://example.invalid/v1/chat/completions",
            ),
        )
        router = AgentLLMRouter(model_suite)
        module = VisualDifferencingModule(router)
        world_state = WorldState(
            current_task="Find the fridge.",
            current_pose="kitchen",
            metadata={
                "sensors": {
                    "rgb": {
                        "data": [
                            [[255, 0, 0], [0, 255, 0]],
                            [[0, 0, 255], [255, 255, 255]],
                        ]
                    }
                }
            },
        )
        trace = LLMCallTrace(
            provider="openai-compatible",
            model="vision-model",
            duration_s=0.01,
            prompt="multimodal",
            response_text='{"summary":"scene from images","reasoning_summary":"vision path","task_completed":false,"change_detected":true,"task_relevant_attributes":["fridge visible"],"delta":[]}',
            raw_response=None,
        )
        with patch.object(
            router,
            "complete_json_messages",
            return_value=(
                {
                    "summary": "scene from images",
                    "reasoning_summary": "vision path",
                    "task_completed": False,
                    "change_detected": True,
                    "task_relevant_attributes": ["fridge visible"],
                    "delta": [],
                },
                trace,
            ),
        ) as mocked:
            observation, observation_trace = module.describe_initial_scene(
                instruction="find the fridge",
                world_state=world_state,
                target="fridge",
            )
        self.assertEqual(observation.summary, "scene from images")
        self.assertIs(observation_trace, trace)
        mocked.assert_called_once()
        messages = mocked.call_args.kwargs["messages"]
        self.assertTrue(any(item.get("type") == "image_url" for item in messages[1]["content"]))


if __name__ == "__main__":
    unittest.main()
