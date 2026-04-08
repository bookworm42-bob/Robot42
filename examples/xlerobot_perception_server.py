from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlerobot_agent.perception_service import PerceptionService, PerceptionServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the XLeRobot perception service. "
            "This service exposes `perceive_scene`, `ground_object_3d`, and "
            "`set_waypoint_from_object` over HTTP for the Ubuntu offload node."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8892)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = PerceptionService(PerceptionServiceConfig(host=args.host, port=args.port))
    print(f"XLeRobot perception service: http://{server.host}:{server.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
