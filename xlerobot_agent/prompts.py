from __future__ import annotations

from .models import GoalContext, SkillContract, WorldState


def build_skill_selection_system_prompt() -> str:
    return (
        "You are the XLeRobot planning module. "
        "You must choose the next skill only from the provided skill registry. "
        "Do not invent new skills. Prefer skills that advance the current goal, "
        "respect feasibility constraints, and are executable by the configured "
        "backend. Return structured reasoning for each scored skill."
    )


def build_skill_selection_user_prompt(
    goal: GoalContext,
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
                "For each feasible skill, estimate two prompt-driven signals: "
                "goal usefulness and success likelihood. Then return the best next skill."
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
