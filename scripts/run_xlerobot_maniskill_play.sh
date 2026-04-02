#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${XLEROBOT_MANISKILL_VENV:-$ROOT_DIR/.venv-maniskill}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  cat <<EOF >&2
Missing ManiSkill environment at $VENV_DIR

Create it first with:
  $ROOT_DIR/scripts/setup_xlerobot_maniskill_env.sh
EOF
  exit 2
fi

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" -m multido_xlerobot.maniskill "$@"
