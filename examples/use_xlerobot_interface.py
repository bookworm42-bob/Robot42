from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multido_xlerobot import XLeRobotBootstrapError, XLeRobotInterface
from multido_xlerobot.bootstrap import DEFAULT_XLEROBOT_FORK_ROOT


def main() -> None:
    api = XLeRobotInterface(DEFAULT_XLEROBOT_FORK_ROOT)
    try:
        print(api.summary())

        robot_config = api.make_robot_config()
        vr_config = api.make_vr_config()

        print(type(robot_config).__name__)
        print(type(vr_config).__name__)
    except XLeRobotBootstrapError as exc:
        print("Bootstrap blocked:", exc)
        print(api.installation_help())


if __name__ == "__main__":
    main()
