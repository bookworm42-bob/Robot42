from __future__ import annotations

from dataclasses import dataclass, field

from .models import SkillContract, SkillType


@dataclass
class SkillRegistry:
    _skills: dict[str, SkillContract] = field(default_factory=dict)

    def register(self, skill: SkillContract) -> None:
        self._skills[skill.skill_id] = skill

    def register_many(self, skills: list[SkillContract]) -> None:
        for skill in skills:
            self.register(skill)

    def get(self, skill_id: str) -> SkillContract:
        return self._skills[skill_id]

    def list_enabled(self) -> list[SkillContract]:
        return [skill for skill in self._skills.values() if skill.enabled]

    def by_type(self, skill_type: SkillType) -> list[SkillContract]:
        return [skill for skill in self.list_enabled() if skill.skill_type == skill_type]

    def skill_ids(self) -> list[str]:
        return list(self._skills.keys())
