from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XLeRobotPaths:
    repo_root: Path
    software_src: Path
    xlevr_root: Path
    record_script: Path
    model_file: Path
    robot_pkg_dir: Path
    robot_2wheels_pkg_dir: Path
    vr_pkg_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "XLeRobotPaths":
        root = Path(repo_root).expanduser().resolve()
        return cls(
            repo_root=root,
            software_src=root / "software" / "src",
            xlevr_root=root / "XLeVR",
            record_script=root / "software" / "src" / "record.py",
            model_file=root / "software" / "src" / "model" / "SO101Robot.py",
            robot_pkg_dir=root / "software" / "src" / "robots" / "xlerobot",
            robot_2wheels_pkg_dir=root / "software" / "src" / "robots" / "xlerobot_2wheels",
            vr_pkg_dir=root / "software" / "src" / "teleporators" / "xlerobot_vr",
        )
