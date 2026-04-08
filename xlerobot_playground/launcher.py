from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def resolve_repo_root(explicit_repo_root: str | None = None) -> Path:
    if explicit_repo_root:
        return Path(explicit_repo_root).expanduser().resolve()
    candidate = Path.home() / "XLeRobot"
    if candidate.exists():
        return candidate
    return Path("/Users/alin/xlerobot_forked")


def default_sim_python_bin(repo_root: Path) -> Path:
    candidate = repo_root / ".venv-maniskill" / "bin" / "python"
    if candidate.exists():
        return candidate
    fallback = Path("/home/alin/Robot42/.venv-maniskill/bin/python")
    if fallback.exists():
        return fallback
    return Path(sys.executable)


def exec_python_module(
    module: str,
    *,
    python_bin: str | Path,
    argv: list[str],
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    command = [str(Path(python_bin).expanduser()), "-m", module, *argv]
    completed = subprocess.run(command, cwd=cwd, env=env)
    return completed.returncode

