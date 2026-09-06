"""用 metadata-only 候选池冻结 V7.2 的小规模干净开发与选路日志。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()

    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    split = yaml.safe_load(args.split.read_text(encoding="utf-8"))
    nuscenes = roles["datasets"]["nuscenes"]
    groups = nuscenes["group_roles"]
    candidates = sorted(map(str, groups["source_candidate_pool"]))
    expected = sorted(
        map(
            str,
            split["roles"]["dev"]
            + split["roles"]["route_select"]
            + split["roles"]["source_test_candidate_unopened"],
        )
    )
    if candidates != expected:
        raise RuntimeError("冻结清单与 metadata-only source candidate pool 不一致")
    if split["quality_used_for_selection"] or split["historical_result_used_for_selection"]:
        raise RuntimeError("clean split 不允许按质量或历史结果选择")

    groups["dev"] = list(split["roles"]["dev"])
    groups["route_select"] = list(split["roles"]["route_select"])
    groups["source_candidate_pool"] = list(
        split["roles"]["source_test_candidate_unopened"]
    )
    groups["source_test"] = []
    nuscenes["frozen_roles"]["route_select"] = True
    nuscenes["frozen_roles"]["source_test"] = False
    nuscenes["notes"] = [
        "The ten dependency-chain-unexposed logs are assigned lexicographically without quality access.",
        "Four logs are development, three are route selection, and three remain unopened source-test candidates.",
        "The reduced log counts are an explicit evidence limitation; adjacent scenes do not enlarge the denominator.",
    ]
    roles["status"] = "clean_dev_and_route_select_frozen_source_test_unopened"
    args.roles.write_text(
        json.dumps(roles, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
