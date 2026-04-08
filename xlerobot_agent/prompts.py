from __future__ import annotations

import json
from typing import Any

from .models import GoalContext, SkillContract, Subgoal, WorldState


def build_instruction_normalization_system_prompt() -> str:
    return (
        "You normalize robot commands into one clean imperative instruction. "
        "Preserve the user intent, remove filler words, and return JSON only with key "
        "`normalized_instruction`."
    )


def build_instruction_normalization_user_prompt(text: str) -> str:
    return f"Raw instruction: {text}"


def build_place_discovery_system_prompt() -> str:
    return (
        "You infer likely places for a household robot from scene evidence and memory. "
        "Return JSON only with key `places`, whose value is a list of objects with keys "
        "`name`, `confidence`, and `evidence`."
    )


def build_place_discovery_user_prompt(world_state: WorldState) -> str:
    return "\n".join(
        [
            f"Current pose: {world_state.current_pose}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Image descriptions: {list(world_state.image_descriptions)}",
            f"Semantic memory summary: {world_state.semantic_memory_summary}",
            f"Spatial memory summary: {world_state.spatial_memory_summary}",
            f"Existing place memories: {[place.name for place in world_state.place_memories]}",
        ]
    )


def build_subgoal_planning_system_prompt() -> str:
    return (
        "You decompose a robot command into a short list of executable subgoals. "
        "Return JSON only with key `subgoals`, whose value is a list of objects with keys "
        "`text`, `kind`, and `target`. Use compact kinds such as `navigate`, `search`, "
        "`align`, `manipulate`, `recover`, or `general`. "
        "Use these patterns abstractly, not literally: "
        "for `go to the kitchen and open the fridge`, prefer subgoals like "
        "`navigate(kitchen) -> search(fridge) -> align(fridge) -> manipulate(fridge)`; "
        "for `map the downstairs area`, prefer a short mapping/exploration-oriented plan such as "
        "`general(downstairs)` instead of low-level tool ids. "
        "Do not emit tool names like `get_map` or `go_to_pose` as subgoal text."
    )


def build_subgoal_planning_user_prompt(
    goal: GoalContext,
    world_state: WorldState,
    skills: list[SkillContract],
) -> str:
    skill_ids = [skill.skill_id for skill in skills]
    return "\n".join(
        [
            f"User goal: {goal.user_instruction}",
            f"Structured goal: {goal.structured_goal or 'none'}",
            f"Current pose: {world_state.current_pose}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Available observations: {sorted(world_state.available_observations)}",
            f"Discovered places: {[place.name for place in world_state.place_memories]}",
            f"Recent execution history: {list(world_state.recent_execution_history)}",
            f"Available skill ids: {skill_ids}",
        ]
    )


def build_skill_selection_system_prompt() -> str:
    return (
        "You are the XLeRobot skill scoring module. "
        "Score every feasible skill only from the provided skill registry. "
        "Do not invent new skills. Return JSON only with key `scores`, whose value is "
        "a list of objects with keys `skill_id`, `goal_usefulness`, `success_likelihood`, "
        "`combined_score`, and `reasoning`. All scores must be between 0 and 1. "
        "Prefer skills for acting on the world once navigation/perception context is sufficient; "
        "prefer tools first when the agent still needs map state, target grounding, or an approach pose."
    )


def build_action_selection_system_prompt() -> str:
    return (
        "You are the planner for an embodied robot agent. "
        "Choose exactly one next action from the provided candidates. "
        "Return JSON only with keys: action_type, action_id, summary, reasoning_summary. "
        "Do not invent action ids that are not in the candidate list. "
        "Use these patterns abstractly, not literally: "
        "named-place navigation usually follows `get_map -> create_map/explore if needed -> go_to_pose`; "
        "object-centric approach usually follows `perceive_scene -> ground_object_3d -> "
        "set_waypoint_from_object -> go_to_pose`; "
        "when the robot is already well positioned and a matching registered VLA skill exists, "
        "prefer `run_vla_skill::<skill_id>` over extra perception or map steps; "
        "if `preferred_action_id` is present and still plausible, bias toward it."
    )


def build_exploration_policy_system_prompt() -> str:
    return (
        "You are the exploration policy for a mobile apartment-mapping robot. "
        "Your job is to choose the next high-level exploration action from validated map-level frontier candidates. "
        "The deterministic stack owns occupancy mapping, frontier deduplication, safety checks, navigation, and scanning. "
        "You must reason about semantic coverage, spatial coverage, and whether the remaining frontier memory still contains useful work. "
        "Return JSON only with keys: decision_type, selected_frontier_id, selected_return_waypoint_id, "
        "frontier_ids_to_store, exploration_complete, reasoning_summary, semantic_updates. "
        "`decision_type` must be one of `explore_frontier`, `revisit_frontier`, or `finish`. "
        "Only use frontier ids and return waypoint ids that already exist in the prompt. "
        "Do not invent raw coordinates. Operate strictly on the provided map-level frontier objects. "
        "Prefer frontiers that likely open new rooms, doors, corridors, or long-range expansions. "
        "Treat sensor-range edge frontiers as real opportunities when the visible map reaches the 10 m sensing limit "
        "and space may continue beyond it. "
        "Keep frontier memory clean: visited and failed frontiers should not be proposed again unless the prompt explicitly "
        "shows them as revalidated. "
        "Set `exploration_complete` to true only when credible exploration opportunities are exhausted. "
        "If there are still reachable stored or candidate frontiers, prefer continuing exploration rather than finishing. "
        "`semantic_updates` must be a list of objects with keys: label, kind, target_id, confidence, evidence. "
        "Use `kind` values such as `region_label`, `sub_area`, or `room_hint`."
    )


def build_exploration_policy_user_prompt(payload: dict[str, Any]) -> str:
    robot = payload.get("robot", {})
    frontier_memory = payload.get("frontier_memory", {})
    sections = [
        "Mission:",
        str(payload.get("mission", "Explore the apartment and finish only when mapping is complete.")),
        "",
        "Robot State:",
        json.dumps(robot, indent=2, sort_keys=True),
        "",
        "Frontier Memory:",
        json.dumps(frontier_memory, indent=2, sort_keys=True),
        "",
        "Candidate Frontiers:",
        json.dumps(payload.get("candidate_frontiers", []), indent=2, sort_keys=True),
        "",
        "Explored Areas:",
        json.dumps(payload.get("explored_areas", []), indent=2, sort_keys=True),
        "",
        "Recent Views:",
        json.dumps(payload.get("recent_views", []), indent=2, sort_keys=True),
        "",
        "Guardrails:",
        json.dumps(payload.get("guardrails", {}), indent=2, sort_keys=True),
        "",
        "ASCII Occupancy Map:",
        "Legend: ? unknown, . free, # occupied, R robot, F active/stored frontier, V visited/completed frontier.",
        str(payload.get("ascii_map", "")),
    ]
    return "\n".join(sections)


def build_skill_selection_user_prompt(
    goal: GoalContext,
    subgoal: Subgoal,
    world_state: WorldState,
    skills: list[SkillContract],
) -> str:
    skill_lines = []
    for skill in skills:
        skill_lines.append(
            "- "
            f"skill_id={skill.skill_id}; "
            f"type={skill.skill_type.value}; "
            f"description={skill.language_description}; "
            f"executor={skill.executor_binding}; "
            f"required_observations={sorted(skill.required_observations)}; "
            f"required_resources={sorted(skill.required_resources)}; "
            f"postcondition={skill.expected_postcondition}"
        )

    return "\n".join(
        [
            f"User goal: {goal.user_instruction}",
            f"Structured goal: {goal.structured_goal or 'none'}",
            f"Current subgoal: {subgoal.text}",
            f"Subgoal kind: {subgoal.kind}",
            f"Subgoal target: {subgoal.target or 'none'}",
            f"Current task: {world_state.current_task}",
            f"Current pose: {world_state.current_pose}",
            f"Localization confidence: {world_state.localization_confidence}",
            f"Visible objects: {sorted(world_state.visible_objects)}",
            f"Visible landmarks: {sorted(world_state.visible_landmarks)}",
            f"Image descriptions: {list(world_state.image_descriptions)}",
            f"Discovered places: {[place.name for place in world_state.place_memories]}",
            f"Available observations: {sorted(world_state.available_observations)}",
            f"Recent execution history: {list(world_state.recent_execution_history)}",
            f"Active resource locks: {sorted(world_state.active_resource_locks)}",
            f"Semantic memory summary: {world_state.semantic_memory_summary}",
            f"Spatial memory summary: {world_state.spatial_memory_summary}",
            "Available skills:",
            *skill_lines,
            (
                "For each feasible skill, estimate goal usefulness and success likelihood, "
                "then compute combined_score. Return one score object per provided skill."
            ),
        ]
    )


def build_voice_command_system_prompt(wake_word: str) -> str:
    return (
        "You normalize spoken robot commands into direct action text for XLeRobot. "
        f"The wake word is `{wake_word}`. Strip the wake word and return the "
        "command in plain imperative English."
    )


def build_voice_command_user_prompt(transcript: str) -> str:
    return f"Transcript: {transcript}"


def build_visual_scene_summary_system_prompt() -> str:
    return (
        "You are the visual differencing module for a household robot. "
        "Summarize the current scene for planning. "
        "Return JSON only with keys: summary, reasoning_summary, task_completed, "
        "change_detected, task_relevant_attributes, delta. "
        "Use task_completed only when the task already appears visually complete."
    )


def build_visual_scene_summary_user_prompt(
    *,
    instruction: str,
    world_state: WorldState,
    scene: dict,
    target: str | None = None,
) -> str:
    payload = {
        "instruction": instruction,
        "target": target,
        "current_pose": world_state.current_pose,
        "visible_objects": sorted(world_state.visible_objects),
        "image_descriptions": list(world_state.image_descriptions),
        "scene": {
            "scene_summary": scene.get("scene_summary"),
            "available_streams": scene.get("available_streams"),
            "query_target": scene.get("query_target"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                    "overlay_text": item.get("overlay_text"),
                }
                for item in scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_visual_difference_system_prompt() -> str:
    return (
        "You are the visual differencing module for a household robot. "
        "Compare the previous and current observations after one action. "
        "Return JSON only with keys: summary, reasoning_summary, task_completed, "
        "change_detected, task_relevant_attributes, delta. "
        "Focus on task-relevant changes and whether the subgoal now appears complete."
    )


def build_visual_difference_user_prompt(
    *,
    instruction: str,
    subgoal: Subgoal,
    previous_world_state: WorldState,
    current_world_state: WorldState,
    previous_scene: dict,
    current_scene: dict,
    action: dict,
) -> str:
    payload = {
        "instruction": instruction,
        "subgoal": {"text": subgoal.text, "kind": subgoal.kind, "target": subgoal.target},
        "action": action,
        "previous_world_state": {
            "current_pose": previous_world_state.current_pose,
            "visible_objects": sorted(previous_world_state.visible_objects),
            "available_observations": sorted(previous_world_state.available_observations),
            "satisfied_preconditions": sorted(previous_world_state.satisfied_preconditions),
            "image_descriptions": list(previous_world_state.image_descriptions),
        },
        "current_world_state": {
            "current_pose": current_world_state.current_pose,
            "visible_objects": sorted(current_world_state.visible_objects),
            "available_observations": sorted(current_world_state.available_observations),
            "satisfied_preconditions": sorted(current_world_state.satisfied_preconditions),
            "image_descriptions": list(current_world_state.image_descriptions),
        },
        "previous_scene": {
            "scene_summary": previous_scene.get("scene_summary"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                }
                for item in previous_scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
        "current_scene": {
            "scene_summary": current_scene.get("scene_summary"),
            "annotations": [
                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "depth_m": item.get("depth_m"),
                }
                for item in current_scene.get("annotations", [])
                if isinstance(item, dict)
            ][:8],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)
