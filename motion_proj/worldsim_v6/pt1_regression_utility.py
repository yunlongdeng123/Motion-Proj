"""V6.4 compiled verified episode 的下游回归缺陷拦截实验。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V6-PT1-REGRESSION-UTILITY-01"
FACTORS = ("actor_states", "trajectories", "semantic_labels", "collision_labels")


class PT1RegressionError(RuntimeError):
    """PT1 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise PT1RegressionError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("actor_id") != actor_id]


def _collisions_without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if actor_id not in row["actor_pair"]]


def _expected_edited(base: Mapping[str, Any], actor_id: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "actor_states": _without_actor(base["actor_states"], actor_id),
        "trajectories": _without_actor(base["trajectories"], actor_id),
        "semantic_labels": _without_actor(base["semantic_labels"], actor_id),
        "collision_labels": _collisions_without_actor(base["collision_labels"], actor_id),
    }


def _arm_accept(arm: str, replay: Mapping[str, Any], expected: Mapping[str, Any], actor_id: str) -> bool:
    if arm == "real_only":
        # Real-only 套件没有反事实 edit 合同，故无法检查依赖陈旧。
        return True
    if arm == "real_plus_naive_synthetic":
        # naive 只检查主 actor state 消失，不检查下游依赖闭包。
        return all(row.get("actor_id") != actor_id for row in replay["actor_states"])
    if arm == "real_plus_v6_verified_compiled":
        return all(_canonical(replay[name]) == _canonical(expected[name]) for name in FACTORS)
    raise PT1RegressionError(f"未知方法臂: {arm}")


def _build_mutations(
    base: Mapping[str, Any], edited: Mapping[str, Any], names: list[str]
) -> dict[str, dict[str, Any]]:
    factor_for_mutation = {
        "stale_actor_states": "actor_states",
        "stale_trajectories": "trajectories",
        "stale_semantic_labels": "semantic_labels",
        "stale_collision_labels": "collision_labels",
    }
    if set(names) != set(factor_for_mutation):
        raise PT1RegressionError("mutation denominator 必须是冻结的四类依赖陈旧")
    result = {}
    for name in names:
        mutated = copy.deepcopy(edited)
        mutated[factor_for_mutation[name]] = copy.deepcopy(base[factor_for_mutation[name]])
        result[name] = mutated
    return result


def _evaluate_once(
    base: Mapping[str, Any], edited: Mapping[str, Any], actor_id: str, mutations: list[str], arms: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = _expected_edited(base, actor_id)
    clean_exact = all(_canonical(edited[name]) == _canonical(expected[name]) for name in FACTORS)
    if not clean_exact:
        raise PT1RegressionError("heldout clean edited replay 不满足冻结 V6 dependency closure")
    cases = _build_mutations(base, edited, mutations)
    verdicts: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for arm in arms:
        clean_accept = _arm_accept(arm, edited, expected, actor_id)
        detected = 0
        false_safe = 0
        for mutation_id in mutations:
            replay = cases[mutation_id]
            harmful = any(_canonical(replay[name]) != _canonical(expected[name]) for name in FACTORS)
            accepted = _arm_accept(arm, replay, expected, actor_id)
            detected += int(not accepted)
            false_safe += int(harmful and accepted)
            verdicts.append(
                {
                    "schema_version": "worldsim_v6.pt1_mutation_verdict.v1",
                    "arm": arm,
                    "mutation_id": mutation_id,
                    "harmful_dependency_regression": harmful,
                    "verdict": "ACCEPT" if accepted else "REJECT",
                    "false_safe": bool(harmful and accepted),
                    "replay_factor_hashes": {name: _content_sha256(replay[name]) for name in FACTORS},
                }
            )
        count = len(mutations)
        metrics[arm] = {
            "clean_accept": clean_accept,
            "mutation_count": count,
            "detected_count": detected,
            "mutation_detection_rate": detected / count,
            "false_safe_count": false_safe,
            "false_safe_rate": false_safe / count,
        }
    return verdicts, metrics


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT1RegressionError("task_id 不匹配")
    if not bool(config["heldout_scenario"].get("scene")):
        raise PT1RegressionError("heldout scenario 未冻结")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__regression-utility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        free_gib = shutil.disk_usage(run_root).free / (1024**3)
        if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT1RegressionError(f"磁盘不足: {free_gib:.2f} GiB")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT1RegressionError("正式运行要求 clean worktree")

        contract_run = _resolve_runs_uri(config["contract_source"]["run"])
        heldout_run = _resolve_runs_uri(config["heldout_scenario"]["run"])
        contract_gate = contract_run / config["contract_source"]["gate_file"]
        heldout_gate = heldout_run / config["heldout_scenario"]["gate_file"]
        heldout_manifest = heldout_run / "MANIFEST.json"
        frozen = [config_path, contract_gate, heldout_gate, heldout_manifest,
                  heldout_run / "BASE_REPLAY.json", heldout_run / "EDITED_REPLAY.json"]
        hashes_before = {str(path): _sha256(path) for path in frozen}
        if _sha256(contract_gate) != config["contract_source"]["gate_sha256"]:
            raise PT1RegressionError("development contract gate hash 漂移")
        if _sha256(heldout_gate) != config["heldout_scenario"]["gate_sha256"]:
            raise PT1RegressionError("heldout gate hash 漂移")
        if _sha256(heldout_manifest) != config["heldout_scenario"]["manifest_sha256"]:
            raise PT1RegressionError("heldout manifest hash 漂移")
        if json.loads(contract_gate.read_text(encoding="utf-8"))["decision"] != "accept_typed_dynamic_edits":
            raise PT1RegressionError("development dependency contract 未接受")
        if json.loads(heldout_gate.read_text(encoding="utf-8"))["decision"] != "accept_same_scene_actor_dependency_sensor_binding":
            raise PT1RegressionError("heldout compiled episode 未接受")

        base = json.loads((heldout_run / "BASE_REPLAY.json").read_text(encoding="utf-8"))
        edited = json.loads((heldout_run / "EDITED_REPLAY.json").read_text(encoding="utf-8"))
        actor_id = str(config["heldout_scenario"]["removed_actor_id"])
        mutations = list(config["mutations"])
        arms = list(config["arms"])
        verdicts1, metrics1 = _evaluate_once(base, edited, actor_id, mutations, arms)
        verdicts2, metrics2 = _evaluate_once(base, edited, actor_id, mutations, arms)
        repeat_exact = _canonical(verdicts1) == _canonical(verdicts2) and _canonical(metrics1) == _canonical(metrics2)
        _write_jsonl(run_dir / "MUTATION_VERDICTS.jsonl", verdicts1)
        _write_json(run_dir / "METHOD_ARMS.json", metrics1)
        _write_json(
            run_dir / "HELDOUT_CONTRACT.json",
            {
                "schema_version": "worldsim_v6.pt1_heldout_contract.v1",
                "development_scene": config["contract_source"]["scene"],
                "heldout_scene": config["heldout_scenario"]["scene"],
                "removed_actor_id": actor_id,
                "base_factor_hashes": {name: _content_sha256(base[name]) for name in FACTORS},
                "clean_edited_factor_hashes": {name: _content_sha256(edited[name]) for name in FACTORS},
                "mutation_ids": mutations,
                "mutation_count": len(mutations),
            },
        )

        wall_seconds = time.monotonic() - started
        gate_cfg = config["gate"]
        checks = {
            "development_contract_accepted": True,
            "heldout_compiled_episode_accepted": True,
            "independent_scene_ids": config["contract_source"]["scene"] != config["heldout_scenario"]["scene"],
            "fixed_mutation_denominator": len(mutations) == int(gate_cfg["expected_mutation_count"]),
            "clean_accept_all_arms": all(row["clean_accept"] for row in metrics1.values()),
            "real_only_false_safe_expected": metrics1["real_only"]["false_safe_rate"] == float(gate_cfg["require_real_only_false_safe_rate"]),
            "naive_false_safe_at_least": metrics1["real_plus_naive_synthetic"]["false_safe_rate"] >= float(gate_cfg["require_naive_false_safe_rate_at_least"]),
            "v6_detection_complete": metrics1["real_plus_v6_verified_compiled"]["mutation_detection_rate"] == float(gate_cfg["require_v6_mutation_detection_rate"]),
            "v6_false_safe_zero": metrics1["real_plus_v6_verified_compiled"]["false_safe_rate"] == float(gate_cfg["require_v6_false_safe_rate"]),
            "repeat_exact": repeat_exact,
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.pt1_regression_utility_gate.v1",
            "checks": checks,
            "decision": "accept_verified_compiled_regression_utility" if checks["passed"] else "reject_pt1_regression_utility",
            "unsupported_metrics": config["unsupported_metrics"],
        }
        _write_json(run_dir / "PT1_REGRESSION_UTILITY_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.pt1_regression_utility_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "development_scene": config["contract_source"]["scene"],
            "heldout_scene": config["heldout_scenario"]["scene"],
            "mutation_count": len(mutations),
            "method_arms": metrics1,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "unsupported_metrics": config["unsupported_metrics"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["HELDOUT_CONTRACT.json", "MUTATION_VERDICTS.jsonl", "METHOD_ARMS.json", "PT1_REGRESSION_UTILITY_GATE.json", "SUMMARY.json"]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.pt1_regression_utility_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
                "frozen_source_sha256": hashes_before,
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {"schema_version": "worldsim_v6.terminal.v1", "status": "blocked", "task_id": TASK_ID,
             "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/pt1_regression_utility_v0.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
