"""V7.2 数据准备入口；P0 先执行角色与合同预检。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--dataset", default="nuscenes")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    roles_path = REPO_ROOT / str(config["data"]["roles"])
    roles = load_data_roles(roles_path)
    group_ids = require_role_access(roles, args.dataset, args.role)
    if not args.validate_only:
        raise RuntimeError("P0 仅开放 --validate-only；v2 cache builder 尚未接入原始数据")
    print(
        json.dumps(
            {
                "status": "validated",
                "dataset": args.dataset,
                "role": args.role,
                "group_count": len(group_ids),
                "target_access": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
